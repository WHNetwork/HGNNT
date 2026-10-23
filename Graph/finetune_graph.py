"""Whole-instance TU classification with optional attributes and node categories."""

import argparse
import random
from pathlib import Path
import torch
from torch import nn

if not __package__:
    # PyCharm Run executes this file directly, without a package context.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .ir import compile_graph
from .utils import (add_model_args, add_training_args, add_transfer_args, make_encoder,
                    seed_all, new_run, CSVLog, write_json, accuracy, save_downstream, summarize,
                    load_encoder_checkpoint)


def load_dataset(a):
    from torch_geometric.datasets import TUDataset
    from sklearn.model_selection import train_test_split
    dataset = TUDataset(str(a.data_dir / "TU"), name=a.dataset, use_node_attr=True)
    # TU node categories (e.g. MUTAG atom types) are legitimate input features.
    # data.y is used only for targets and stratified splitting, never in data.x.
    feature_dim = dataset.num_node_features if a.use_node_features else 0
    if a.use_node_features and feature_dim <= 0:
        raise ValueError(f"{a.dataset} has no node features; omit --use_node_features for structure-only mode")
    samples, labels = [], []
    for data in dataset:
        program = compile_graph(data.num_nodes, data.edge_index)
        if a.use_node_features:
            if data.x is None or data.x.ndim != 2 or data.x.shape != (data.num_nodes, feature_dim):
                raise ValueError("TU node features must have one row per node and a consistent feature dimension")
            features = data.x.clone()
        else:
            features = None
        samples.append((program, features))
        labels.append(int(data.y.item()))
    indices = list(range(len(samples)))
    train, held = train_test_split(indices, test_size=a.val_ratio+a.test_ratio,
                                   stratify=labels, random_state=a.split_seed)
    val, test = train_test_split(held, test_size=a.test_ratio/(a.val_ratio+a.test_ratio),
        stratify=[labels[i] for i in held], random_state=a.split_seed+1)
    return samples, labels, dict(train=train, val=val, test=test), feature_dim, dataset.num_classes


def run_seed(a, seed, run, source):
    seed_all(seed)
    samples, labels, splits, feature_dim, classes = source
    encoder = make_encoder(a, feature_dim or None, seed).to(a.device)
    head = nn.Linear(encoder.hidden_dim, classes).to(a.device)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()),
                                  lr=a.learning_rate, weight_decay=a.weight_decay)
    write_json(run / "config.json", dict(vars(a), seed=seed, split_indices=splits, feature_dim=feature_dim))
    rng = random.Random(seed)
    def logits_for(index):
        program, x = samples[index]
        # Every forward encodes ONE graph. No attention can cross graph boundaries.
        return head(encoder(program, x).graph_embedding).unsqueeze(0)
    def evaluate(indices):
        encoder.eval(); head.eval()
        with torch.no_grad():
            logits = torch.cat([logits_for(i) for i in indices])
            target = torch.tensor([labels[i] for i in indices], device=a.device)
            return accuracy(logits, target)
    best, best_epoch = -1., 0
    with CSVLog(run / "epochs.csv", ("dataset", "mode", "seed", "epoch", "train_loss",
                                     "validation_accuracy", "best_epoch")) as log:
        for epoch in range(1, a.epochs+1):
            encoder.train(); head.train()
            order = list(splits["train"])
            rng.shuffle(order)
            total = 0.
            for i in order:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(logits_for(i), torch.tensor([labels[i]], device=a.device))
                loss.backward(); optimizer.step()
                total += loss.item()
            val = evaluate(splits["val"])
            if val > best:
                best, best_epoch = val, epoch
                save_downstream(run / "best.pt", encoder, head, a, seed, epoch, val,
                                 split_indices=splits, feature_dim=feature_dim)
            log.write(dict(dataset=a.dataset, mode=a.mode, seed=seed, epoch=epoch,
                           train_loss=total/len(order), validation_accuracy=val, best_epoch=best_epoch))
    ckpt = torch.load(run / "best.pt", map_location=a.device, weights_only=True)
    load_encoder_checkpoint(encoder, ckpt)
    head.load_state_dict(ckpt["head_state_dict"])
    test = evaluate(splits["test"])
    return dict(dataset=a.dataset, mode=a.mode, seed=seed, best_epoch=best_epoch,
                validation_accuracy=best, test_accuracy=test)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p); add_training_args(p, "graph"); add_transfer_args(p)
    p.add_argument("--dataset", default="MUTAG")
    p.add_argument("--use_node_features", action="store_true",
                   help="use full TU data.x: continuous attributes and encoded node categories; default is structure-only")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=.1)
    p.add_argument("--test_ratio", type=float, default=.1)
    a = p.parse_args(argv)
    if a.val_ratio <= 0 or a.test_ratio <= 0 or a.val_ratio+a.test_ratio >= 1:
        p.error("positive validation/test ratios must sum to less than one")
    source = load_dataset(a)
    root = new_run(a.output_dir)
    results = [run_seed(a, s, new_run(root / f"seed_{s}"), source) for s in (a.seeds or [a.seed])]
    summarize(root, results, ("validation_accuracy", "test_accuracy"))


if __name__ == "__main__":
    main()
