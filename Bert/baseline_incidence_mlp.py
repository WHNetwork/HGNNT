"""MLP on concat(X, X) or concat(X, degree-normalized incidence propagation)."""

import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import Tensor, nn

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Bert"

from .finetune_node_cls import REPO_ROOT, _single_datasets_, load_node_dataset
from .incidence_rewire import rewire_incidence


@torch.no_grad()
def propagate_features(x: Tensor, incidences: Tensor, num_hyperedges: int) -> Tensor:
    """Compute D_v^-1 H D_e^-1 H^T X using only sparse incidence indices.

    Empty edges and isolated nodes produce zero vectors. x contains raw node
    features; this function never receives labels or classifier predictions.
    """
    nodes, edges = incidences.to(x.device)
    edge_sizes = torch.bincount(edges, minlength=num_hyperedges).to(x.dtype)
    node_degrees = torch.bincount(nodes, minlength=x.shape[0]).to(x.dtype)
    edge_features = x.new_zeros((num_hyperedges, x.shape[1]))
    edge_features.index_add_(0, edges, x.index_select(0, nodes))
    edge_features /= edge_sizes.clamp_min(1).unsqueeze(-1)
    propagated = torch.zeros_like(x)
    propagated.index_add_(0, nodes, edge_features.index_select(0, edges))
    return propagated / node_degrees.clamp_min(1).unsqueeze(-1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=_single_datasets_, default="cora")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature_mode", choices=("features_only", "original", "rewired"),
                        default="features_only")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rewire_seed", type=int, default=42)
    parser.add_argument("--rewire_multiplier", type=int, default=5)
    parser.add_argument("--output_dir", type=Path)
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.rewire_multiplier < 0:
        parser.error("rewire_multiplier must be nonnegative")
    if args.output_dir is None:
        args.output_dir = (REPO_ROOT / "Bert" / "runs" /
                           f"incidence_mlp_{args.dataset}_{args.feature_mode}_seed{args.seed}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if any((args.output_dir / name).exists() for name in ("results.csv", "best_model.pt")):
        raise FileExistsError(f"Training outputs already exist in {args.output_dir}; use a new output_dir")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, labels, masks = load_node_dataset(args.dataset, args.seed)
    if any(not mask.any().item() for mask in masks.values()):
        raise ValueError("train, valid and test splits must each contain labeled nodes")

    # Compute the fixed input once on CPU, before constructing the classifier.
    x = data.x.to(device="cpu", dtype=torch.float32)
    structure_stats = None
    if args.feature_mode == "features_only":
        propagated = x
    else:
        # Same bipartite V->E extraction and edge-ID offset as the HIR entry.
        num_nodes = x.shape[0]
        index = data.hyperedge_index.cpu()
        vertex_to_edge = (index[0] < num_nodes) & (index[1] >= num_nodes)
        incidences = index[:, vertex_to_edge].clone().long()
        incidences[1] -= num_nodes
        # Binary H: repeated source memberships count only once.
        incidences = torch.unique(incidences, dim=1)
        inferred_edges = int(incidences[1].max()) + 1 if incidences.shape[1] else 0
        num_edges = max(int(data.data.num_hyperedges), inferred_edges)
        if args.feature_mode == "rewired":
            incidences, structure_stats = rewire_incidence(
                incidences, num_nodes, num_edges, seed=args.rewire_seed,
                multiplier=args.rewire_multiplier)
            print(" ".join(f"{key}={structure_stats[key]}" for key in (
                "num_nodes", "num_hyperedges", "num_incidences", "successful_swaps",
                "changed_incidence_ratio")))
            print(f"target_swaps={structure_stats['target_swaps']} "
                  f"attempts={structure_stats['attempts']} "
                  f"stop_reason={structure_stats['stop_reason']}")
        propagated = propagate_features(x, incidences, num_edges)
    inputs = torch.cat((x, propagated), dim=-1).to(args.device)
    del data, x, propagated
    num_classes = int(labels.max()) + 1
    labels = labels.to(args.device)
    masks = {split: mask.to(args.device) for split, mask in masks.items()}

    # Reset after all loading and preprocessing for identical initialization
    # and initial dropout RNG state in all three modes.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = nn.Sequential(
        nn.Linear(inputs.shape[1], 128), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_valid, best_epoch, best_test = -1.0, 0, 0.0
    print(f"feature_mode={args.feature_mode}; test_acc reports the best validation checkpoint so far")

    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("dataset", "feature_mode", "seed", "epoch", "train_loss",
                  "train_accuracy", "valid_accuracy", "test_accuracy", "best_epoch",
                  "is_best", "rewire_seed", "rewire_multiplier")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits[masks["train"]], labels[masks["train"]])
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            del logits, loss

            model.eval()
            with torch.no_grad():
                predictions = model(inputs).argmax(dim=-1)
                accuracy = {split: (predictions[masks[split]] == labels[masks[split]])
                            .float().mean().item() for split in ("train", "valid")}
                is_best = accuracy["valid"] > best_valid
                if is_best:
                    best_valid, best_epoch = accuracy["valid"], epoch
                    best_test = (predictions[masks["test"]] == labels[masks["test"]]).float().mean().item()
            # Test results never participate in checkpoint selection.
            if is_best:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": inputs.shape[1],
                    "num_classes": num_classes,
                    "dataset": args.dataset,
                    "feature_mode": args.feature_mode,
                    "seed": args.seed,
                    "epoch": epoch,
                    "learning_rate": args.learning_rate,
                    "rewire_seed": args.rewire_seed,
                    "rewire_multiplier": args.rewire_multiplier,
                    "structure_stats": structure_stats,
                    "split_masks": {split: mask.cpu() for split, mask in masks.items()},
                    "valid_accuracy": best_valid,
                    "test_accuracy": best_test,
                }, checkpoint_path)
            writer.writerow(dict(dataset=args.dataset, feature_mode=args.feature_mode,
                                 seed=args.seed, epoch=epoch, train_loss=train_loss,
                                 train_accuracy=accuracy["train"], valid_accuracy=accuracy["valid"],
                                 test_accuracy=best_test, best_epoch=best_epoch, is_best=is_best,
                                 rewire_seed=args.rewire_seed, rewire_multiplier=args.rewire_multiplier))
            file.flush()
            print(f"epoch={epoch} loss={train_loss:.4f} "
                  f"valid_acc={accuracy['valid']:.4f} test_acc={best_test:.4f}")
    print(f"feature_mode={args.feature_mode} best_epoch={best_epoch} "
          f"best_valid_accuracy={best_valid:.4f} best_test_accuracy={best_test:.4f}")


if __name__ == "__main__":
    main()
