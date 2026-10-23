"""Structure-only masked COMPLETE-edge reconstruction and held-out evaluation."""

import argparse
import random
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm

if not __package__:
    # PyCharm Run executes this file directly, without a package context.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .data import DATA_ROOT, FAMILIES, SPLITS, load_graph, sample_masked_edges, split_paths
from .encoder import GraphIREncoder
from .utils import (add_model_args, add_training_args, model_config, seed_all,
                    new_run, CSVLog, write_json, binary_metrics, PairHead, load_encoder_checkpoint)

TRAIN_FAMILIES = ("ba", "ws", "sbm")
METRICS = ("loss", "auc", "ap", "graphs", "skipped_graphs", "actual_mask_ratio")


def family_mask_seeds(seed):
    return {family: seed + len(SPLITS) * i for i, family in enumerate(FAMILIES)}


class MaskedEdgePretrainer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = PairHead(encoder.hidden_dim)

    def forward(self, batch):
        nodes = self.encoder(batch.program).node_embeddings
        pairs = torch.cat((batch.positive_pairs, batch.negative_pairs)).to(nodes.device)
        logits = self.head(nodes, pairs)
        labels = torch.cat((torch.ones(len(batch.positive_pairs), device=nodes.device),
                            torch.zeros(len(batch.negative_pairs), device=nodes.device)))
        return logits, labels


def evaluate_family(model, paths, ratio, seed):
    """Pool predictions within ONE family, resetting masks/negatives each call."""
    rng = random.Random(seed)
    model.eval()
    loss_sum, pairs, graphs, skipped, hidden, original = 0., 0, 0, 0, 0, 0
    scores, targets = [], []
    with torch.no_grad():
        for path in tqdm(paths, desc="evaluate", leave=False):
            batch = sample_masked_edges(load_graph(path), ratio, rng)
            if batch is None:
                skipped += 1
                continue
            logits, labels = model(batch)
            loss_sum += nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum").item()
            pairs += len(labels)
            graphs += 1
            hidden += len(batch.positive_pairs)
            original += batch.original_edges
            scores.append(logits.cpu())
            targets.append(labels.cpu())
    if not pairs:
        return dict(loss=None, auc=None, ap=None, graphs=0,
                    skipped_graphs=skipped, actual_mask_ratio=None)
    return dict(loss=loss_sum/pairs, **binary_metrics(torch.cat(targets), torch.cat(scores)),
                graphs=graphs, skipped_graphs=skipped, actual_mask_ratio=hidden/max(original, 1))


def evaluate(model, paths, ratio, seed, required_families=()):
    """Separate family pools and equal-weight macro metrics, never pooled scores."""
    groups = {family: [] for family in FAMILIES}
    for path in paths:
        groups[path.parent.name].append(path)
    seeds = family_mask_seeds(seed)
    families = {family: evaluate_family(model, groups[family], ratio, seeds[family])
                for family in FAMILIES}
    for family in required_families:
        if not families[family]["graphs"]:
            raise ValueError(f"Validation family {family.upper()} has no valid masked edge samples; "
                             "BA/WS/SBM checkpoint selection requires all three families")

    def macro(selected):
        # Missing families leave the macro undefined; never change its weights.
        return {metric: (sum(families[f][metric] for f in selected) / len(selected)
                         if all(families[f]["graphs"] for f in selected) else None)
                for metric in ("loss", "auc", "ap")}

    return dict(families=families, macro_train=macro(TRAIN_FAMILIES), macro_all=macro(FAMILIES))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_training_args(p, "pretrain", epochs=50)
    p.add_argument("--data_dir", type=Path, default=DATA_ROOT / "synthetic_v1")
    p.add_argument("--mask_ratio", type=float, default=.15)
    p.add_argument("--max_train_graphs", type=int)
    p.add_argument("--max_val_graphs", type=int)
    a = p.parse_args(argv)
    if not 0 < a.mask_ratio <= 1:
        p.error("mask_ratio must be in (0,1]")
    seed_all(a.seed)
    paths = {s: split_paths(a.data_dir, s,
                a.max_train_graphs if s == "train" else a.max_val_graphs if s == "val" else None,
                a.seed, families=TRAIN_FAMILIES if s == "train" else FAMILIES) for s in SPLITS}
    if any(not paths[s] for s in SPLITS):
        raise ValueError("data_dir needs nonempty train/val/test/ood splits")
    run = new_run(a.output_dir)
    metadata = {s: [str(path.relative_to(a.data_dir)) for path in values] for s, values in paths.items()}
    family_config = dict(train=TRAIN_FAMILIES, val=FAMILIES, test=FAMILIES, ood=FAMILIES)
    mask_seeds = {split: family_mask_seeds(a.seed + offset)
                  for split, offset in (("val", 1), ("test", 2), ("ood", 3))}
    selection = dict(metric="validation_macro_loss", families=TRAIN_FAMILIES,
                     reduction="equal_weight_mean", mode="min")
    protocol = dict(family_config=family_config, data_split_metadata=metadata,
                    mask_seeds=mask_seeds, checkpoint_selection=selection,
                    held_out_generation_family="er", ood_dimension="node_count")
    experiment_config = {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(a).items()}
    write_json(run / "config.json", dict(experiment_config, **protocol))
    model = MaskedEdgePretrainer(GraphIREncoder(**model_config(a))).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    rng = random.Random(a.seed)
    best = float("inf")
    fields = ("epoch", "train_loss", "validation_macro_loss", "train_graphs",
              "actual_mask_ratio", "train_skipped_graphs") + tuple(
                  f"validation_{family}_{metric}" for family in FAMILIES for metric in ("loss", "auc", "ap"))
    with CSVLog(run / "epochs.csv", fields) as log, CSVLog(
            run / "validation_families.csv", ("epoch", "family", *METRICS)) as family_log:
        for epoch in range(1, a.epochs+1):
            model.train()
            order = list(paths["train"])
            rng.shuffle(order)
            loss_sum, pairs, graphs, skipped, hidden, original = 0., 0, 0, 0, 0, 0
            for path in tqdm(order, desc=f"epoch {epoch}"):
                batch = sample_masked_edges(load_graph(path), a.mask_ratio, rng)
                if batch is None:
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                logits, labels = model(batch)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                loss.backward()
                optimizer.step()
                loss_sum += loss.detach().item() * len(labels)
                pairs += len(labels)
                graphs += 1
                hidden += len(batch.positive_pairs)
                original += batch.original_edges
            if not pairs:
                raise ValueError("no training graphs have valid masked edge samples")
            val = evaluate(model, paths["val"], a.mask_ratio, a.seed+1,
                           required_families=TRAIN_FAMILIES)
            macro_loss = val["macro_train"]["loss"]
            family_metrics = {f"validation_{family}_{metric}": metrics[metric]
                              for family, metrics in val["families"].items()
                              for metric in ("loss", "auc", "ap")}
            log.write(dict(epoch=epoch, train_loss=loss_sum/pairs,
                validation_macro_loss=macro_loss, train_graphs=graphs,
                actual_mask_ratio=hidden/max(original, 1), train_skipped_graphs=skipped,
                **family_metrics))
            for family, metrics in val["families"].items():
                family_log.write(dict(epoch=epoch, family=family, **metrics))
            print(f"epoch={epoch} train_loss={loss_sum/pairs:.5f} "
                  f"validation_macro_loss={macro_loss:.5f} skipped={skipped}")
            if macro_loss < best:
                best = macro_loss
                torch.save(dict(encoder_config=model.encoder.config,
                    encoder_state_dict=model.encoder.state_dict(), pretraining_head_state_dict=model.head.state_dict(),
                    epoch=epoch, seed=a.seed, validation_metric=val,
                    validation_macro_loss=macro_loss, experiment_config=experiment_config,
                    mask_ratio=a.mask_ratio, validation_mask_seed=a.seed+1,
                    **protocol), run / "best.pt")
    checkpoint = torch.load(run / "best.pt", map_location=a.device, weights_only=True)
    load_encoder_checkpoint(model.encoder, checkpoint)
    model.head.load_state_dict(checkpoint["pretraining_head_state_dict"])
    results, detailed_results = {}, {}
    for split, seed in (("test", a.seed+2), ("ood", a.seed+3)):
        metrics = evaluate(model, paths[split], a.mask_ratio, seed)
        detailed_results[split] = metrics
        for family, values in metrics["families"].items():
            results.update({f"{split}_{family}_{key}": value for key, value in values.items()})
        for group in ("macro_train", "macro_all"):
            results.update({f"{split}_{group}_{key}": value for key, value in metrics[group].items()})
    with CSVLog(run / "evaluation.csv", tuple(results)) as log:
        log.write(results)
    write_json(run / "evaluation.json", dict(best_epoch=checkpoint["epoch"],
        validation_macro_loss=checkpoint["validation_macro_loss"], results=detailed_results, **protocol))
    print(results)


if __name__ == "__main__":
    main()
