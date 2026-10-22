"""Planetoid public-split node classification; retrain original/rewired graphs."""

import argparse
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
                    seed_all, new_run, CSVLog, write_json, load_planetoid,
                    rewire_graph, accuracy, save_downstream, summarize)


def run_seed(a, seed, run):
    seed_all(seed)
    data, classes = load_planetoid(a.data_dir, a.dataset)
    program = compile_graph(data.num_nodes, data.edge_index)
    stats = dict(successful_swaps=0, changed_edge_ratio=0., structure="original")
    if a.structure == "rewired":
        program, stats = rewire_graph(program, a.rewire_seed, a.swap_multiplier)
    masks = {name: getattr(data, f"{name}_mask").to(a.device) for name in ("train", "val", "test")}
    if any(not m.any().item() for m in masks.values()):
        raise ValueError("public masks must all contain labeled nodes")
    x, y = data.x.to(a.device), data.y.to(a.device)
    seed_all(seed)
    encoder = make_encoder(a, x.shape[1], seed).to(a.device)
    head = nn.Linear(encoder.hidden_dim, classes).to(a.device)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()),
                                  lr=a.learning_rate, weight_decay=a.weight_decay)
    write_json(run / "config.json", dict(vars(a), seed=seed, structure_stats=stats,
                                         protocol="retrain_and_evaluate_on_same_structure"))
    best, best_epoch = -1., 0
    fields = ("dataset", "mode", "structure", "seed", "epoch", "train_loss", "train_accuracy",
              "validation_accuracy", "test_accuracy", "best_epoch")
    with CSVLog(run / "epochs.csv", fields) as log:
        for epoch in range(1, a.epochs+1):
            encoder.train(); head.train()
            optimizer.zero_grad(set_to_none=True)
            logits = head(encoder(program, x).node_embeddings)
            loss = nn.functional.cross_entropy(logits[masks["train"]], y[masks["train"]])
            loss.backward(); optimizer.step()
            encoder.eval(); head.eval()
            with torch.no_grad():
                logits = head(encoder(program, x).node_embeddings)
                train = accuracy(logits[masks["train"]], y[masks["train"]])
                val = accuracy(logits[masks["val"]], y[masks["val"]])
            if val > best:
                best, best_epoch = val, epoch
                save_downstream(run / "best.pt", encoder, head, a, seed, epoch, val,
                    split_masks={k: v.cpu() for k, v in masks.items()}, structure_stats=stats,
                    edge_index=program.to_edge_index(), train_loss=loss.item(), train_accuracy=train)
            # Test is deliberately unavailable during model selection.
            log.write(dict(dataset=a.dataset, mode=a.mode, structure=a.structure, seed=seed, epoch=epoch,
                train_loss=loss.item(), train_accuracy=train, validation_accuracy=val,
                test_accuracy="", best_epoch=best_epoch))
    ckpt = torch.load(run / "best.pt", map_location=a.device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"]); head.load_state_dict(ckpt["head_state_dict"])
    encoder.eval(); head.eval()
    with torch.no_grad():
        logits = head(encoder(program, x).node_embeddings)
        test = accuracy(logits[masks["test"]], y[masks["test"]])
    result = dict(dataset=a.dataset, mode=a.mode, seed=seed,
                  epoch=best_epoch, train_loss=ckpt["train_loss"], train_accuracy=ckpt["train_accuracy"],
                  validation_accuracy=best, test_accuracy=test, best_epoch=best_epoch, **stats)
    write_json(run / "result.json", result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p)
    add_training_args(p, "node")
    add_transfer_args(p)
    p.add_argument("--dataset", choices=("Cora", "Citeseer"), default="Cora")
    p.add_argument("--structure", choices=("original", "rewired"), default="original")
    p.add_argument("--rewire_seed", type=int, default=42)
    p.add_argument("--swap_multiplier", type=float, default=5.)
    a = p.parse_args(argv)
    root = new_run(a.output_dir)
    results = [run_seed(a, seed, new_run(root / f"seed_{seed}")) for seed in (a.seeds or [a.seed])]
    summarize(root, results, ("validation_accuracy", "test_accuracy"))


if __name__ == "__main__":
    main()
