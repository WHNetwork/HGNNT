"""Independent GraphGPT-HIR SMTP pretraining (requires transformers==4.53.3)."""

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

if not __package__:
    # Direct file execution (including IDE Run) needs the repository package root.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .data import DATA_ROOT, FAMILIES, SPLITS, load_graph, split_paths
from .graphgpt_hir_model import GraphGPTHIR
from .graphgpt_hir_tokenizer import GraphGPTHIRTokenizer
from .ir import compile_graph
from .utils import CSVLog, add_training_args, new_run, positive_int, seed_all, write_json

TRAIN_FAMILIES = ("ba", "ws", "sbm")
STAT_FIELDS = ("smtp_loss", "identity_accuracy", "masked_tokens", "identity_tokens",
               "masked_nodes", "nodes", "actual_mask_ratio", "actual_node_mask_ratio",
               "mean_scheduled_ratio", "graphs", "skipped_graphs")
IDENTIFIABILITY_LIMIT = (
    "With no node attributes, jointly hiding every occurrence of several node identities "
    "can make their labels/occurrence assignments non-identifiable. First-appearance "
    "order provides numbering priors; SMTP accuracy is not unique structural recovery."
)
SOURCES = dict(
    paper="https://arxiv.org/abs/2401.00529v3",
    graphgpt="https://github.com/alibaba/graph-gpt",
    masking="src/data/tokenizer/masking.py::_mask_stacked_input_ids / _get_mask_ratio",
    backbone="src/models/graphgpt/utils_graphgpt.py::LlamaModel",
    embedding="src/models/graphgpt/modeling_common.py::StackedFeatAggregation",
    loss="src/models/graphgpt/modeling_helpers.py::_get_ce_loss",
    transformers_version="4.53.3",
)


def run_paths(model, tokenizer, paths, rng, data_root, sequence_metadata, optimizer=None):
    """One full graph per step; only unmasked raw GraphIR is passed to the adapter."""
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    loss_sum = correct = masked = identities = masked_nodes = nodes = graphs = skipped = 0
    ratio_sum = 0.
    with torch.enable_grad() if training else torch.no_grad():
        for path in tqdm(paths, desc="train SMTP" if training else "evaluate SMTP", leave=False):
            graph = load_graph(path)
            program = compile_graph(graph.num_nodes, graph.edge_index)
            try:
                sequence = tokenizer.encode(program, rng)
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            # Retain exact length and inverse maps of the latest encoded view.
            # These records are checkpoint/log metadata, NEVER model inputs.
            sequence_metadata[str(path.relative_to(data_root))] = dict(
                token_length=sequence.token_length, num_nodes=sequence.num_nodes,
                num_edges=program.num_edges, node_mapping=sequence.node_mapping,
                edge_order=sequence.edge_order)
            example = tokenizer.mask(sequence, rng)
            if not example.masked_tokens:
                skipped += 1
                continue
            inputs, labels = example.input_ids.to(device), example.labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(inputs, labels=labels)
            if training:
                output.loss.backward()
                optimizer.step()
            mask = labels != -100
            correct += ((output.logits.detach().argmax(-1) == labels) & mask).sum().item()
            loss_sum += output.loss.detach().item() * example.masked_tokens
            masked += example.masked_tokens
            identities += example.identity_tokens
            masked_nodes += example.masked_nodes
            nodes += sequence.num_nodes
            ratio_sum += example.scheduled_ratio
            graphs += 1
    return dict(smtp_loss=loss_sum / masked if masked else None,
                identity_accuracy=correct / masked if masked else None,
                masked_tokens=masked, identity_tokens=identities, masked_nodes=masked_nodes,
                nodes=nodes, actual_mask_ratio=masked / identities if identities else None,
                actual_node_mask_ratio=masked_nodes / nodes if nodes else None,
                mean_scheduled_ratio=ratio_sum / graphs if graphs else None,
                graphs=graphs, skipped_graphs=skipped)


def evaluate(model, tokenizer, paths, seed, data_root, sequence_metadata, required_families=()):
    groups = {family: [] for family in FAMILIES}
    for path in paths:
        groups[path.parent.name].append(path)
    families = {family: run_paths(model, tokenizer, groups[family],
                                  random.Random(seed + len(SPLITS) * i), data_root, sequence_metadata)
                for i, family in enumerate(FAMILIES)}
    for family in required_families:
        if not families[family]["masked_tokens"]:
            raise ValueError(f"Validation family {family.upper()} has no SMTP targets; "
                             "BA/WS/SBM model selection requires all three families")

    def macro(selected):
        return {key: (sum(families[f][key] for f in selected) / len(selected)
                      if all(families[f]["masked_tokens"] for f in selected) else None)
                for key in ("smtp_loss", "identity_accuracy")}

    return dict(families=families, macro_train=macro(TRAIN_FAMILIES), macro_all=macro(FAMILIES))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_args(parser, "graphgpt_hir_pretrain", epochs=50)
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT / "synthetic_v1")
    parser.add_argument("--hidden_dim", type=positive_int, default=256)
    parser.add_argument("--num_layers", type=positive_int, default=4)
    parser.add_argument("--num_heads", type=positive_int, default=4)
    parser.add_argument("--intermediate_size", type=positive_int, default=1024)
    parser.add_argument("--attention_dropout", type=float, default=0.)
    parser.add_argument("--identity_vocab_size", type=positive_int, default=2048)
    parser.add_argument("--max_length", type=positive_int, default=8192)
    parser.add_argument("--max_train_graphs", type=positive_int)
    parser.add_argument("--max_val_graphs", type=positive_int)
    args = parser.parse_args(argv)
    seed_all(args.seed)
    paths = {split: split_paths(args.data_dir, split,
                                args.max_train_graphs if split == "train" else
                                args.max_val_graphs if split == "val" else None,
                                args.seed, families=TRAIN_FAMILIES if split == "train" else FAMILIES)
             for split in SPLITS}
    if any(not values for values in paths.values()):
        raise ValueError("data_dir needs nonempty train/val/test/ood selections")
    tokenizer = GraphGPTHIRTokenizer(args.identity_vocab_size, args.max_length)
    model = GraphGPTHIR(vocab_size=tokenizer.vocab_size, max_length=args.max_length,
                        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                        num_heads=args.num_heads, intermediate_size=args.intermediate_size,
                        attention_dropout=args.attention_dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    run = new_run(args.output_dir)
    split_metadata = {split: [str(path.relative_to(args.data_dir)) for path in values]
                      for split, values in paths.items()}
    mask_seeds = {split: {family: args.seed + offset + len(SPLITS)*i
                          for i, family in enumerate(FAMILIES)}
                  for split, offset in (("val", 1), ("test", 2), ("ood", 3))}
    protocol = dict(
        family_config=dict(train=TRAIN_FAMILIES, val=FAMILIES, test=FAMILIES, ood=FAMILIES),
        data_split_metadata=split_metadata, evaluation_view_mask_seeds=mask_seeds,
        training_view_mask_seed=args.seed,
        checkpoint_selection=dict(metric="validation_macro_loss", families=TRAIN_FAMILIES,
                                  reduction="equal_weight_mean", mode="min"),
        smtp_schedule=dict(name="linear", time_distribution="uniform_0_1", ratio="1-t",
                           mask_unit="node_identity_all_occurrences", replacement="mask_only",
                           supervision="identity_column_only", loss="unweighted_cross_entropy"),
        held_out_generation_family="er", ood_dimension="node_count",
        identifiability_limit=IDENTIFIABILITY_LIMIT, sources=SOURCES)
    experiment_config = {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()}
    write_json(run / "config.json", dict(experiment_config=experiment_config,
        model_config=model.config, llama_config=model.backbone.config.to_dict(),
        tokenizer_config=tokenizer.config, **protocol))
    print(IDENTIFIABILITY_LIMIT)
    rng = random.Random(args.seed)
    sequence_metadata = {}
    best = float("inf")
    fields = ("epoch", "validation_macro_loss", "validation_macro_accuracy") + tuple(
        f"train_{key}" for key in STAT_FIELDS) + tuple(
        f"validation_{family}_{key}" for family in FAMILIES for key in ("smtp_loss", "identity_accuracy"))
    with CSVLog(run / "epochs.csv", fields) as log, CSVLog(
            run / "validation_families.csv", ("epoch", "family", *STAT_FIELDS)) as family_log:
        for epoch in range(1, args.epochs + 1):
            order = list(paths["train"])
            rng.shuffle(order)
            train = run_paths(model, tokenizer, order, rng, args.data_dir, sequence_metadata, optimizer)
            if not train["masked_tokens"]:
                raise ValueError("training selection has no node identity targets")
            val = evaluate(model, tokenizer, paths["val"], args.seed+1, args.data_dir,
                           sequence_metadata, required_families=TRAIN_FAMILIES)
            macro_loss = val["macro_train"]["smtp_loss"]
            row = dict(epoch=epoch, validation_macro_loss=macro_loss,
                       validation_macro_accuracy=val["macro_train"]["identity_accuracy"])
            row.update({f"train_{key}": value for key, value in train.items()})
            for family, values in val["families"].items():
                family_log.write(dict(epoch=epoch, family=family, **values))
                row.update({f"validation_{family}_{key}": values[key]
                            for key in ("smtp_loss", "identity_accuracy")})
            log.write(row)
            print(f"epoch={epoch} train_smtp_loss={train['smtp_loss']:.6f} "
                  f"validation_macro_loss={macro_loss:.6f}")
            if macro_loss < best:
                best = macro_loss
                torch.save(dict(model_config=model.config, tokenizer_config=tokenizer.config,
                    model_state_dict=model.state_dict(), llama_config=model.backbone.config.to_dict(),
                    epoch=epoch, seed=args.seed, validation_metrics=val, validation_macro_loss=macro_loss,
                    sequence_metadata=sequence_metadata, training_rng_state=rng.getstate(),
                    experiment_config=experiment_config, **protocol), run / "best.pt")
    checkpoint = torch.load(run / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    results, flat = {}, {}
    for split, offset in (("test", 2), ("ood", 3)):
        metrics = evaluate(model, tokenizer, paths[split], args.seed+offset, args.data_dir, sequence_metadata)
        results[split] = metrics
        for family, values in metrics["families"].items():
            flat.update({f"{split}_{family}_{key}": value for key, value in values.items()})
        for group in ("macro_train", "macro_all"):
            flat.update({f"{split}_{group}_{key}": value for key, value in metrics[group].items()})
    with CSVLog(run / "evaluation.csv", tuple(flat)) as log:
        log.write(flat)
    write_json(run / "evaluation.json", dict(best_epoch=checkpoint["epoch"],
        validation_macro_loss=checkpoint["validation_macro_loss"], results=results, **protocol))
    # Final records contain latest training views and fixed validation/test/OOD views.
    write_json(run / "sequence_metadata.json", sequence_metadata)
    print(flat)


if __name__ == "__main__":
    main()
