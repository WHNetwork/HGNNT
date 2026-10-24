"""HIR Structural Pretraining. Run directly or with -m Graph.sequence_pretrain."""

import argparse
from contextlib import nullcontext
import csv
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .data import DATA_ROOT
from .sequence_encoder import ARCHITECTURE_VERSION, INITIALIZATION_VERSION, HIRSequenceEncoder
from .sequence_pretrain_data import prepare_corpus, sample_node_pairs, second_view
from .sequence_pretrain_objectives import (LOSS_NAMES, EpochMetrics, StructuralHeads, class_weights,
    combine_losses, contrastive_loss, cosine_schedule, supervised_losses, view_loss)

PRETRAINING_VERSION = "hir_sp_v1"


def encode_batch(encoder, programs):
    """Reuse V2 modules, with bidirectional valid-key mask and no padded readout.

    Padding queries are unused, but have valid keys to avoid all-masked rows.
    The explicit additive 4D mask bypasses Llama's causal-mask builder.
    """
    sequences = [encoder.tokenizer(p) for p in programs]
    embedded = [encoder.embed(s, x=None) for s in sequences]
    states = pad_sequence(embedded, batch_first=True)
    batch, length = states.shape[:2]
    lengths = torch.tensor([s.token_length for s in sequences], device=states.device)
    positions = torch.arange(length, device=states.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    mask = states.new_zeros((batch, 1, 1, length))
    mask.masked_fill_(~valid[:, None, None, :], torch.finfo(states.dtype).min)
    mask = mask.expand(batch, 1, length, length)
    hidden = encoder.backbone(inputs_embeds=states, attention_mask=mask,
        position_ids=positions.unsqueeze(0).expand(batch, -1), use_cache=False,
        return_dict=True).last_hidden_state
    return [encoder.readout(s, hidden[i, :s.token_length]) for i, s in enumerate(sequences)]


def batches(items, batch_size, token_budget, rng=None, views=2):
    order = list(items)
    if rng is not None:
        rng.shuffle(order)
    batch, longest = [], 0
    for item in order:
        length = 5*item.program.num_edges + sum(item.graph.degree(u) == 0 for u in item.graph) + 1
        if views*length > token_budget:
            raise ValueError(f"{item.path}: {views} view(s) exceed --max_batch_tokens; increase explicit budget")
        if batch and (len(batch) >= batch_size or views*max(longest, length)*(len(batch)+1) > token_budget):
            yield batch
            batch, longest = [], 0
        batch.append(item)
        longest = max(longest, length)
    if batch:
        yield batch


def compute_batch(encoder, heads, items, args, rng, weights, meter):
    views, mappings, samples = [], [], []
    cross_view = args.lambda_contrast > 0 or args.lambda_view > 0
    for item in items:
        # Consume the same two seeds even for disabled tasks. Toggling a view
        # ablation must not implicitly change node-pair supervision.
        pair_rng = random.Random(rng.randrange(2**63))
        view_rng = random.Random(rng.randrange(2**63))
        if cross_view:
            view, order = second_view(item.program, view_rng)
            views.append(view)
            mappings.append(order)
        pairs, labels, candidates = (sample_node_pairs(item, args.pairs_per_graph, pair_rng)
                                     if args.lambda_node > 0 else ([], [], [0]*4))
        samples.append((pairs, labels))
        meter.node_candidate_counts = [a+b for a, b in zip(meter.node_candidate_counts, candidates)]
    programs = [i.program for i in items] + (views if cross_view else [])
    outputs = encode_batch(encoder, programs)
    first, second = outputs[:len(items)], outputs[len(items):]
    collected = {name: [] for name in LOSS_NAMES}
    graph_active = []
    for row, (item, output, sample) in enumerate(zip(items, first, samples)):
        losses = supervised_losses(heads, output, item, *sample, weights, meter, args.coefficients)
        meter.add_representation(output)
        if args.lambda_view > 0:
            other = second[row]
            inverse = sorted(range(len(mappings[row])), key=mappings[row].__getitem__)
            losses["node_view"] = view_loss(output.node_embeddings, other.node_embeddings)
            losses["edge_view"] = view_loss(output.edge_embeddings, other.edge_embeddings[inverse])
            meter.valid_samples["node_view"] += len(output.node_embeddings)
            meter.valid_samples["edge_view"] += len(output.edge_embeddings)
        active, _ = combine_losses(losses, args.coefficients)
        graph_active.append(active is not None)
        for name, value in losses.items():
            if value is not None:
                collected[name].append(value)
                meter.add_loss(name, value)
    means = {name: torch.stack(values).mean() if values else None for name, values in collected.items()}
    if args.lambda_contrast > 0:
        means["contrast"], negatives = contrastive_loss(heads, [o.graph_embedding for o in first],
            [o.graph_embedding for o in second], items, args.temperature)
        meter.negative_counts.extend(negatives)
        meter.valid_samples["contrast"] += sum(n > 0 for n in negatives)
        meter.add_loss("contrast", means["contrast"])
        graph_active = [active or count > 0 for active, count in zip(graph_active, negatives)]
    meter.skipped_graphs += sum(not active for active in graph_active)
    total, groups = combine_losses(means, args.coefficients)
    for name in ("edge", "graph", "view"):
        meter.add_loss(name, groups[name])
    meter.add_loss("total", total)
    return total


def autocast_context(args):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.effective_bf16 else nullcontext()


def epoch_pass(encoder, heads, items, args, epoch, weights, optimizer=None, scheduler=None):
    training = optimizer is not None
    encoder.train(training)
    heads.train(training)
    meter = EpochMetrics()
    # Validation seed does not depend on epoch. Augmentation and pair sampling
    # have independent streams from model/dropout RNG and training order.
    seed = args.seed + (1000003*(epoch+1) if training else 2000000011)
    rng = random.Random(seed)
    view_count = 2 if args.lambda_contrast > 0 or args.lambda_view > 0 else 1
    iterator = batches(items, args.batch_size, args.max_batch_tokens, rng if training else None, views=view_count)
    pending, updates = 0, 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    def update(count):
        # Correct the last partial accumulation window, including skipped batches.
        if count != args.grad_accumulation:
            for parameter in list(encoder.parameters()) + list(heads.parameters()):
                if parameter.grad is not None:
                    parameter.grad.mul_(args.grad_accumulation/count)
        nn.utils.clip_grad_norm_(list(encoder.parameters())+list(heads.parameters()), args.gradient_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    with torch.enable_grad() if training else torch.no_grad():
        for batch in iterator:
            with autocast_context(args):
                loss = compute_batch(encoder, heads, batch, args, rng, weights if training else None, meter)
            if loss is None:
                meter.skipped_batches += 1
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError("HIR-SP total loss is nonfinite")
            if training:
                (loss/args.grad_accumulation).backward()
                pending += 1
                if pending == args.grad_accumulation:
                    update(pending)
                    pending = 0
                    updates += 1
        if training and pending:
            update(pending)
            updates += 1
    return meter.result(), updates


def dump_json(path, value):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def rng_state():
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("resume requires the saved CUDA RNG device count")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, encoder, heads, optimizer, scheduler, epoch, step, best, args, manifest, validation):
    checkpoint = encoder.checkpoint_dict()
    checkpoint.update(pretraining_version=PRETRAINING_VERSION, pretraining_head_state_dict=heads.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
        epoch=epoch, global_step=step, best_validation_loss=best, seed=args.seed,
        objective_config=dict(coefficients=args.coefficients, temperature=args.temperature,
                              pairs_per_graph=args.pairs_per_graph, projection_dim=args.projection_dim),
        training_config=args.saved_config, data_manifest=manifest, rng_state=rng_state(),
        validation_metrics=validation, selection_metric="equal-family mean validation total loss",
        weight_source="HIR-SP training", resume_boundary="end_of_epoch")
    torch.save(checkpoint, path)


def export_encoder(path, encoder, epoch, manifest, validation):
    checkpoint = encoder.checkpoint_dict()
    checkpoint.update(pretraining_version=PRETRAINING_VERSION, weight_source="HIR-SP trained encoder",
                      epoch=epoch, data_manifest=manifest, validation_metrics=validation)
    torch.save(checkpoint, path)


def flatten(value, prefix=""):
    out = {}
    for name, item in value.items():
        key = f"{prefix}_{name}" if prefix else name
        if isinstance(item, dict):
            out.update(flatten(item, key))
        else:
            out[key] = json.dumps(item) if isinstance(item, list) else item
    return out


def append_csv(path, row):
    exists = path.exists()
    if exists:
        with path.open(encoding="utf-8", newline="") as stream:
            fields = next(csv.reader(stream))
        if set(fields) != set(row):
            raise ValueError("epoch CSV schema does not match this run")
    else:
        fields = list(row)
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def family_metric_macro(validation, families):
    """Equal-family metrics without pooling predictions across generators."""
    result = {}
    for task in ("node", "triangle", "bridge", "components", "graph_triangle", "bipartite"):
        result[task] = {}
        for metric in ("accuracy", "macro_accuracy", "macro_f1", "f1", "ap"):
            values = [validation[f][task][metric] for f in families if validation[f][task][metric] is not None]
            result[task][metric] = sum(values)/len(values) if values else None
            result[task][metric+"_valid_families"] = len(values)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", type=Path, default=DATA_ROOT/"synthetic_v1")
    p.add_argument("--output_dir", type=Path, default=Path(__file__).resolve().parent/"runs"/"hir_sp")
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--size", choices=("small", "base"), default="base")
    p.add_argument("--id_encoding", choices=("hex", "lookup"), default="hex")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accumulation", type=int, default=1)
    p.add_argument("--max_graphs", type=int)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--max_batch_tokens", type=int, default=32768)
    p.add_argument("--pairs_per_graph", type=int, default=128)
    p.add_argument("--projection_dim", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("adamw", "adam"), default="adamw")
    p.add_argument("--scheduler", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--weight_decay", type=float, default=.01)
    p.add_argument("--warmup_ratio", type=float, default=.05)
    p.add_argument("--gradient_clip", type=float, default=1.)
    p.add_argument("--temperature", type=float, default=.1)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    for name, default in dict(node=1., edge=1., graph=1., contrast=.1, view=.1).items():
        p.add_argument(f"--lambda_{name}", type=float, default=default)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    integers = (args.epochs, args.batch_size, args.grad_accumulation, args.max_length,
                args.max_batch_tokens, args.pairs_per_graph, args.projection_dim, args.save_every)
    args.coefficients = {name: getattr(args, "lambda_"+name) for name in ("node", "edge", "graph", "contrast", "view")}
    if (min(integers) < 1 or args.seed < 0 or args.temperature <= 0 or args.learning_rate <= 0
            or args.gradient_clip <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio < 1
            or any(v < 0 or not math.isfinite(v) for v in args.coefficients.values())
            or not any(args.coefficients.values()) or (args.max_graphs is not None and args.max_graphs < 1)):
        p.error("invalid counts, optimization parameters, or loss coefficients")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use --device cpu explicitly")
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        args.effective_bf16 = bool(args.bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    args.saved_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                         if k not in ("resume", "output_dir", "saved_config")}
    args.saved_config["sampling_protocol"] = "bounded_bfs_independent_pair_view_rng_v1"
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("pretraining_version") != PRETRAINING_VERSION:
            raise ValueError("resume accepts only complete HIR-SP v1 checkpoints")
        if checkpoint.get("training_config") != args.saved_config:
            raise ValueError("resume configuration differs; reuse original training arguments including --epochs")
        encoder = HIRSequenceEncoder.from_checkpoint(args.resume, device)
    else:
        checkpoint = None
        encoder = HIRSequenceEncoder(size=args.size, id_encoding=args.id_encoding,
                                     max_length=args.max_length).to(device)
    if encoder.config["feature_dim"] is not None:
        raise ValueError("HIR-SP requires feature_dim=None and x=None")
    if (encoder.config["architecture_version"] != ARCHITECTURE_VERSION
            or encoder.config["initialization_version"] != INITIALIZATION_VERSION):
        raise ValueError("incompatible sequence encoder")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.output_dir/name).exists() for name in ("epochs.csv", "last.pt", "best.pt")):
        raise FileExistsError("run already exists; use --resume or a new --output_dir")
    dump_json(args.output_dir/"config.json", args.saved_config)
    print(f"Preparing existing corpus: {args.data_dir.resolve()} (labels and exact isomorphism deduplication)", flush=True)
    records, manifest = prepare_corpus(args.data_dir, args.seed, args.max_graphs, args.max_length)
    if checkpoint and checkpoint["data_manifest"] != manifest:
        raise ValueError("data split/label manifest changed since the checkpoint")
    weights, counts = class_weights(records["train"])
    _, val_counts = class_weights(records["val"])
    coverage = dict(train=counts, val=val_counts,
                    missing_classes={s: {t: [c for c, n in enumerate(v) if n == 0] for t, v in dist.items()}
                                     for s, dist in (("train", counts), ("val", val_counts))},
                    note="Absent classes cannot be learned/evaluated; counts do not establish adequate generalization coverage.")
    dump_json(args.output_dir/"data_manifest.json", manifest)
    dump_json(args.output_dir/"label_coverage.json", coverage)
    print(f"HIR-SP {args.size}/{args.id_encoding}, BF16={args.effective_bf16}; label coverage: {coverage}", flush=True)
    heads = StructuralHeads(encoder.hidden_dim, args.projection_dim).to(device)
    optimizer_class = torch.optim.AdamW if args.optimizer == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(list(encoder.parameters())+list(heads.parameters()),
                                lr=args.learning_rate, weight_decay=args.weight_decay)
    # Epoch permutations can alter token packing. Plan the exact deterministic
    # packing count without executing a model; skipped batches may reduce updates.
    view_count = 2 if args.lambda_contrast > 0 or args.lambda_view > 0 else 1
    total_steps = sum(math.ceil(sum(1 for _ in batches(records["train"], args.batch_size, args.max_batch_tokens,
        random.Random(args.seed+1000003*(epoch+1)), views=view_count))/args.grad_accumulation)
        for epoch in range(args.epochs))
    def schedule(step):
        if args.scheduler == "cosine":
            return cosine_schedule(step, total_steps, args.warmup_ratio)
        return min(1., (step+1)/max(1, int(total_steps*args.warmup_ratio)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start, global_step, best = 0, 0, float("inf")
    if checkpoint:
        heads.load_state_dict(checkpoint["pretraining_head_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, global_step, best = checkpoint["epoch"]+1, checkpoint["global_step"], checkpoint["best_validation_loss"]
        restore_rng(checkpoint["rng_state"])
    families = sorted({i.family for i in records["val"]})
    known_generators = any(i.family in ("ba", "ws", "sbm", "er") for i in records["train"])
    selection = ["ba", "ws", "sbm"] if known_generators else families
    for family in selection:
        if family not in families:
            raise ValueError(f"missing checkpoint-selection validation family: {family}")
    for epoch in range(start, args.epochs):
        train, updates = epoch_pass(encoder, heads, records["train"], args, epoch, weights, optimizer, scheduler)
        if not updates:
            dump_json(args.output_dir/"no_effective_training_epoch.json", dict(epoch=epoch, train=train))
            raise RuntimeError("all training batches lacked an enabled valid objective; no trained checkpoint exported")
        global_step += updates
        validation = {}
        for family in families:
            validation[family], _ = epoch_pass(encoder, heads, [i for i in records["val"] if i.family == family],
                                               args, epoch, weights)
        if any(validation[f]["total_loss"] is None for f in selection):
            raise ValueError("a checkpoint-selection family has no effective validation objective")
        macro = sum(validation[f]["total_loss"] for f in selection)/len(selection)
        row = dict(epoch=epoch, global_step=global_step, learning_rate=optimizer.param_groups[0]["lr"],
                   val_total_loss=macro, validation_selection_families=selection)
        row.update(flatten(train, "train"))
        row.update(flatten(validation, "val_family"))
        metric_macro = family_metric_macro(validation, selection)
        row.update(flatten(metric_macro, "val_macro"))
        # Explicit family-macro loss columns; no cross-family prediction pooling.
        for loss in LOSS_NAMES + ("edge", "graph", "view"):
            values = [validation[f][loss+"_loss"] for f in selection if validation[f][loss+"_loss"] is not None]
            row["val_"+loss+"_loss"] = sum(values)/len(values) if values else None
        append_csv(args.output_dir/"epochs.csv", row)
        with (args.output_dir/"metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(epoch=epoch, train=train, validation=validation,
                validation_macro_loss=macro, validation_metric_macro=metric_macro,
                selection_families=selection), allow_nan=False)+"\n")
        improved = macro < best
        best = min(best, macro)
        state = dict(macro_loss=macro, families=validation, selection_families=selection)
        save_checkpoint(args.output_dir/"last.pt", encoder, heads, optimizer, scheduler,
                        epoch, global_step, best, args, manifest, state)
        if improved:
            save_checkpoint(args.output_dir/"best.pt", encoder, heads, optimizer, scheduler,
                            epoch, global_step, best, args, manifest, state)
            export_encoder(args.output_dir/"encoder_best.pt", encoder, epoch, manifest, state)
        if (epoch+1) % args.save_every == 0:
            save_checkpoint(args.output_dir/f"epoch_{epoch+1:04d}.pt", encoder, heads, optimizer, scheduler,
                            epoch, global_step, best, args, manifest, state)
        print(f"epoch={epoch} train={train['total_loss']} val_macro={macro:.6f} steps={global_step} "
              f"skipped_batches={train['skipped_batches']}", flush=True)
    export_encoder(args.output_dir/"encoder_last.pt", encoder, max(start-1, args.epochs-1), manifest,
                   dict(best_validation_loss=best))


if __name__ == "__main__":
    main()
