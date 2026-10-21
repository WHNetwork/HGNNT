"""Two-layer feature-only MLP using the HIR-BERT node classification split."""

import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Bert"

from .finetune_node_cls import REPO_ROOT, _single_datasets_, load_node_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=_single_datasets_, default="cora")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=Path,
                        default=REPO_ROOT / "Bert" / "runs" / "mlp_cora_seed0")
    args = parser.parse_args(argv)
    if args.epochs < 1:
        parser.error("epochs must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data, labels, masks = load_node_dataset(args.dataset, args.seed)
    if any(not mask.any().item() for mask in masks.values()):
        raise ValueError("train, valid and test splits must each contain labeled nodes")
    x = data.x.to(device=args.device, dtype=torch.float32)
    num_classes = int(labels.max()) + 1
    labels = labels.to(args.device)
    masks = {split: mask.to(args.device) for split, mask in masks.items()}
    del data

    model = nn.Sequential(
        nn.Linear(x.shape[1], 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_valid, best_epoch, best_test = -1.0, 0, 0.0

    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        fields = ("dataset", "seed", "epoch", "train_loss", "train_accuracy",
                  "valid_accuracy", "test_accuracy", "is_best")
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits[masks["train"]], labels[masks["train"]])
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            del logits, loss

            model.eval()
            with torch.no_grad():
                predictions = model(x).argmax(dim=-1)
                accuracy = {split: (predictions[mask] == labels[mask]).float().mean().item()
                            for split, mask in masks.items()}
            # Match HIR-BERT: select strictly by validation accuracy.
            is_best = accuracy["valid"] > best_valid
            if is_best:
                best_valid, best_epoch, best_test = accuracy["valid"], epoch, accuracy["test"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": x.shape[1],
                    "num_classes": num_classes,
                    "dataset": args.dataset,
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
                                 test_accuracy=accuracy["test"], is_best=is_best))
            file.flush()
            print(f"epoch={epoch} train_loss={train_loss:.4f} "
                  f"train_accuracy={accuracy['train']:.4f} "
                  f"valid_accuracy={accuracy['valid']:.4f} "
                  f"test_accuracy={accuracy['test']:.4f}")
    print(f"best_epoch={best_epoch} best_valid_accuracy={best_valid:.4f} "
          f"best_test_accuracy={best_test:.4f}")


if __name__ == "__main__":
    main()
