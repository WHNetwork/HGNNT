"""Supervised HIR-BERT node classification with optional DHG-Bench node features."""

import argparse
import csv
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_ROOT = REPO_ROOT / "dhgbench"
sys.path.insert(0, str(BENCH_ROOT))
if not __package__:
    sys.path.insert(0, str(REPO_ROOT))
    __package__ = "Bert"

from lib_dataset import _single_datasets_
from lib_dataset.data_base import HyperDataset

from .encoder import HIRBertEncoder, HIRGuidedEncoder
from .hir import compile_hypergraph
from .incidence_rewire import rewire_incidence


class FeatureResidual(nn.Module):
    """Direct feature path plus a learnable contribution from H-IR context."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, hir_embeddings):
        return self.feature_projection(x) + self.alpha * hir_embeddings


def load_node_dataset(dataset: str, seed: int):
    """Shared raw features, labels and benchmark split; no H-IR compilation."""
    # The existing loader resolves ../data relative to the benchmark directory.
    loader_args = SimpleNamespace(dname=dataset, method="HIRBert", device="cpu",
                                  feature_noise="0.6")
    previous_dir = Path.cwd()
    try:
        os.chdir(BENCH_ROOT)
        data = HyperDataset(loader_args)
    finally:
        os.chdir(previous_dir)

    num_nodes = data.x.shape[0]
    data.num_nodes = num_nodes
    masks = data.generate_random_split(train_ratio=0.5, val_ratio=0.25, seed=seed)
    labels = data.y.long().reshape(-1)
    # Retain the benchmark split while excluding any unlabeled nodes.
    masks = {split: mask & (labels >= 0) for split, mask in masks.items()}
    return data, labels, masks


def load_node_data(dataset: str, seed: int, incidence_mode: str = "original",
                   rewire_seed: int = 42, rewire_multiplier: int = 5):
    data, labels, masks = load_node_dataset(dataset, seed)
    num_nodes = data.num_nodes

    # Loader indices contain both V->E and E->V, with edge IDs offset by N.
    index = data.hyperedge_index
    vertex_to_edge = (index[0] < num_nodes) & (index[1] >= num_nodes)
    incidences = index[:, vertex_to_edge].clone().long()
    incidences[1] -= num_nodes
    inferred_edges = int(incidences[1].max()) + 1 if incidences.shape[1] else 0
    num_edges = max(int(data.data.num_hyperedges), inferred_edges)
    num_incidences = len(set(map(tuple, incidences.t().tolist())))
    structure_stats = dict(num_nodes=num_nodes, num_hyperedges=num_edges,
                          num_incidences=num_incidences, target_swaps=0,
                          successful_swaps=0, attempts=0, max_attempts=0,
                          changed_incidence_ratio=0.0, stop_reason="original structure")
    if incidence_mode == "rewired":
        incidences, structure_stats = rewire_incidence(
            incidences, num_nodes, num_edges, seed=rewire_seed, multiplier=rewire_multiplier)
    program = compile_hypergraph(num_nodes, incidences, num_hyperedges=num_edges)
    return program, data.x, labels, masks, structure_stats


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=_single_datasets_, default="cora")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder_type", choices=("v2", "guided"), default="v2")
    parser.add_argument("--incidence_mode", choices=("original", "rewired"), default="original")
    parser.add_argument("--rewire_seed", type=int, default=42)
    parser.add_argument("--rewire_multiplier", type=int, default=5)
    parser.add_argument("--use_node_features", action="store_true",
                        help="Inject dataset node features into NODE_REF embeddings")
    parser.add_argument("--feature_residual", action="store_true",
                        help="Fuse Linear(X) + learnable alpha * HIR node embeddings")
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.rewire_multiplier < 0:
        parser.error("rewire_multiplier must be nonnegative")
    if args.output_dir is None:
        args.output_dir = REPO_ROOT / "Bert" / "runs" / f"node_cls_{args.incidence_mode}"
    if args.feature_residual and not args.use_node_features:
        parser.error("--feature_residual requires --use_node_features")
    if args.encoder_type == "guided" and args.feature_residual:
        parser.error("--feature_residual is only supported with --encoder_type v2")
    return args


def main(argv=None):
    args = parse_args(argv)
    # Keep the two ablation runs from overwriting each other's output files.
    if any((args.output_dir / name).exists() for name in ("results.csv", "best_model.pt")):
        raise FileExistsError(f"Training outputs already exist in {args.output_dir}; use a new output_dir")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    program, x, labels, masks, structure_stats = load_node_data(
        args.dataset, args.seed, args.incidence_mode, args.rewire_seed, args.rewire_multiplier)
    print(f"dataset={args.dataset} encoder_type={args.encoder_type} "
          f"incidence_mode={args.incidence_mode} rewire_seed={args.rewire_seed}")
    print(" ".join(f"{key}={structure_stats[key]}" for key in (
        "num_nodes", "num_hyperedges", "num_incidences", "successful_swaps",
        "changed_incidence_ratio")))
    if args.incidence_mode == "rewired":
        print(f"successful_swaps={structure_stats['successful_swaps']}/"
              f"{structure_stats['target_swaps']} attempts={structure_stats['attempts']} "
              f"stop_reason={structure_stats['stop_reason']}")
    if any(not mask.any().item() for mask in masks.values()):
        raise ValueError("train, valid and test splits must each contain labeled nodes")
    num_classes = int(labels.max()) + 1
    labels = labels.to(args.device)
    masks = {split: mask.to(args.device) for split, mask in masks.items()}

    feature_dim = x.shape[1] if args.use_node_features else None
    x = x.to(args.device) if args.use_node_features else None
    # Reset after loading/compilation so both structures start with identical
    # encoder and classifier parameters, independent of loader RNG consumption.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder_class = HIRGuidedEncoder if args.encoder_type == "guided" else HIRBertEncoder
    encoder = encoder_class(feature_dim=feature_dim).to(args.device)
    classifier = nn.Linear(encoder.hidden_dim, num_classes).to(args.device)
    feature_residual = (FeatureResidual(feature_dim, encoder.hidden_dim).to(args.device)
                        if args.feature_residual else None)
    parameters = list(encoder.parameters()) + list(classifier.parameters())
    if feature_residual is not None:
        parameters += list(feature_residual.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_valid, best_epoch, best_test = -1.0, 0, 0.0

    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("dataset", "seed", "epoch", "train_loss", "train_accuracy",
                  "valid_accuracy", "test_accuracy", "is_best", "alpha",
                  "incidence_mode", "rewire_seed", "rewire_multiplier")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            encoder.train()
            classifier.train()
            optimizer.zero_grad()
            node_embeddings = encoder(program, x=x).node_embeddings
            if feature_residual is not None:
                node_embeddings = feature_residual(x, node_embeddings)
            logits = classifier(node_embeddings)
            loss = criterion(logits[masks["train"]], labels[masks["train"]])
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            del logits, loss, node_embeddings

            encoder.eval()
            classifier.eval()
            with torch.no_grad():
                node_embeddings = encoder(program, x=x).node_embeddings
                if feature_residual is not None:
                    node_embeddings = feature_residual(x, node_embeddings)
                predictions = classifier(node_embeddings).argmax(dim=-1)
                accuracy = {split: (predictions[mask] == labels[mask]).float().mean().item()
                            for split, mask in masks.items()}
            alpha = feature_residual.alpha.item() if feature_residual is not None else None
            # Selection uses validation accuracy only; test is reported, not optimized.
            is_best = accuracy["valid"] > best_valid
            if is_best:
                best_valid, best_epoch, best_test = accuracy["valid"], epoch, accuracy["test"]
                torch.save({
                    "encoder_type": args.encoder_type,
                    "incidence_mode": args.incidence_mode,
                    "rewire_seed": args.rewire_seed,
                    "rewire_multiplier": args.rewire_multiplier,
                    "structure_stats": structure_stats,
                    "encoder_config": encoder.config,
                    "encoder_state_dict": encoder.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "feature_residual": args.feature_residual,
                    "feature_residual_state_dict": (feature_residual.state_dict()
                                                    if feature_residual is not None else None),
                    "num_classes": num_classes,
                    "dataset": args.dataset,
                    "use_node_features": args.use_node_features,
                    "seed": args.seed,
                    "epoch": epoch,
                    "learning_rate": args.learning_rate,
                    "split_masks": {split: mask.cpu() for split, mask in masks.items()},
                    "valid_accuracy": best_valid,
                    "test_accuracy": best_test,
                }, checkpoint_path)
            writer.writerow(dict(dataset=args.dataset, seed=args.seed, epoch=epoch,
                                 train_loss=train_loss, train_accuracy=accuracy["train"],
                                 valid_accuracy=accuracy["valid"],
                                 test_accuracy=accuracy["test"], is_best=is_best, alpha=alpha,
                                 incidence_mode=args.incidence_mode, rewire_seed=args.rewire_seed,
                                 rewire_multiplier=args.rewire_multiplier))
            file.flush()
            print(f"epoch={epoch} loss={train_loss:.4f} "
                  f"valid_acc={accuracy['valid']:.4f} test_acc={accuracy['test']:.4f}"
                  + (f" alpha={alpha:.6f}" if alpha is not None else ""))
    print(f"best_epoch={best_epoch} best_valid_accuracy={best_valid:.4f} "
          f"best_test_accuracy={best_test:.4f}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
