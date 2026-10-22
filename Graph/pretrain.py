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

from .data import DATA_ROOT, SPLITS, load_graph, sample_masked_edges, split_paths
from .encoder import GraphIREncoder
from .utils import (add_model_args, add_training_args, model_config, seed_all,
                    new_run, CSVLog, write_json, binary_metrics, PairHead)


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


def evaluate(model, paths, ratio, seed):
    """Reset the dedicated sampling RNG: fixed masks AND negatives across epochs."""
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
        raise ValueError("split has no graphs with valid masked positive/negative pairs")
    return dict(loss=loss_sum/pairs, **binary_metrics(torch.cat(targets), torch.cat(scores)),
                graphs=graphs, skipped_graphs=skipped, actual_mask_ratio=hidden/max(original, 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p, max_edges=512)
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
                a.seed) for s in SPLITS}
    if any(not paths[s] for s in SPLITS):
        raise ValueError("data_dir needs nonempty train/val/test/ood splits")
    run = new_run(a.output_dir)
    metadata = {s: [str(path.relative_to(a.data_dir)) for path in values] for s, values in paths.items()}
    write_json(run / "config.json", dict(vars(a), data_split_metadata=metadata,
        mask_seeds=dict(validation=a.seed+1, test=a.seed+2, ood=a.seed+3)))
    model = MaskedEdgePretrainer(GraphIREncoder(**model_config(a))).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    rng = random.Random(a.seed)
    best = float("inf")
    fields = ("epoch", "train_loss", "validation_loss", "validation_auc", "validation_ap",
              "train_graphs", "validation_graphs", "actual_mask_ratio", "validation_actual_mask_ratio",
              "train_skipped_graphs", "validation_skipped_graphs")
    with CSVLog(run / "epochs.csv", fields) as log:
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
            val = evaluate(model, paths["val"], a.mask_ratio, a.seed+1)
            log.write(dict(epoch=epoch, train_loss=loss_sum/pairs,
                validation_loss=val["loss"], validation_auc=val["auc"], validation_ap=val["ap"],
                train_graphs=graphs, validation_graphs=val["graphs"], actual_mask_ratio=hidden/max(original, 1),
                validation_actual_mask_ratio=val["actual_mask_ratio"], train_skipped_graphs=skipped,
                validation_skipped_graphs=val["skipped_graphs"]))
            print(f"epoch={epoch} train_loss={loss_sum/pairs:.5f} validation_loss={val['loss']:.5f} skipped={skipped}")
            if val["loss"] < best:
                best = val["loss"]
                torch.save(dict(encoder_config=model.encoder.config,
                    encoder_state_dict=model.encoder.state_dict(), pretraining_head_state_dict=model.head.state_dict(),
                    epoch=epoch, seed=a.seed, data_split_metadata=metadata, validation_metric=val,
                    mask_ratio=a.mask_ratio, validation_mask_seed=a.seed+1), run / "best.pt")
    checkpoint = torch.load(run / "best.pt", map_location=a.device, weights_only=True)
    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.head.load_state_dict(checkpoint["pretraining_head_state_dict"])
    results = {}
    for split, seed in (("test", a.seed+2), ("ood", a.seed+3)):
        metrics = evaluate(model, paths[split], a.mask_ratio, seed)
        results.update({f"{split}_{key}": value for key, value in metrics.items()})
    with CSVLog(run / "evaluation.csv", tuple(results)) as log:
        log.write(results)
    print(results)


if __name__ == "__main__":
    main()
