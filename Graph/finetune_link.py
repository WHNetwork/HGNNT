"""Reproducible undirected link prediction with disjoint held-out connections."""

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

from .data import pairs_index, sample_non_edges
from .ir import compile_graph
from .utils import (add_model_args, add_training_args, add_transfer_args, make_encoder,
                    seed_all, new_run, CSVLog, write_json, load_planetoid, PairHead,
                    binary_metrics, save_downstream, summarize)


def link_split(program, seed, val_ratio=.1, test_ratio=.1):
    if val_ratio <= 0 or test_ratio <= 0 or val_ratio+test_ratio >= 1:
        raise ValueError("positive validation/test ratios must sum to less than one")
    edges = [e.endpoints for e in program.instructions]
    rng = random.Random(seed)
    rng.shuffle(edges)
    nv, nt = max(1, int(len(edges)*val_ratio)), max(1, int(len(edges)*test_ratio))
    if nv+nt >= len(edges):
        raise ValueError("not enough edges for nonempty train/val/test")
    positives = dict(val=edges[:nv], test=edges[nv:nv+nt], train=edges[nv+nt:])
    negatives = sample_non_edges(program.num_nodes, edges, len(edges), rng)
    rng.shuffle(negatives)
    negative_groups = dict(val=negatives[:nv], test=negatives[nv:nv+nt], train=negatives[nv+nt:])
    splits = {}
    for name, positive in positives.items():
        splits[name] = (pairs_index(positive + negative_groups[name]).t(),
                        torch.cat((torch.ones(len(positive)), torch.zeros(len(positive)))))
    visible = compile_graph(program.num_nodes, pairs_index(positives["train"]))
    return visible, splits


def sample_training_edges(train_program, negative_pairs, ratio, rng):
    """Hide complete training targets; negatives come from link_split's absent pairs."""
    if not 0 < ratio <= 1:
        raise ValueError("train_supervision_ratio must be in (0,1]")
    edges = [e.endpoints for e in train_program.instructions]
    if not edges:
        raise ValueError("link training requires at least one training positive edge")
    count = max(1, round(len(edges) * ratio))
    if len(negative_pairs) < count:
        raise ValueError("not enough training nonedges for balanced supervision")
    selected = set(rng.sample(range(len(edges)), count))
    supervision = [edge for i, edge in enumerate(edges) if i in selected]
    message = [edge for i, edge in enumerate(edges) if i not in selected]
    # Even a one-edge training graph is valid: its message graph then has no edges.
    program = compile_graph(train_program.num_nodes, pairs_index(message))
    pairs = pairs_index(supervision + rng.sample(negative_pairs, count)).t()
    labels = torch.cat((torch.ones(count), torch.zeros(count)))
    return program, pairs, labels


def run_seed(a, seed, run):
    seed_all(seed)
    data, _ = load_planetoid(a.data_dir, a.dataset)
    # Split RNG is independent of model mode, initialization, and training RNG.
    visible, splits = link_split(compile_graph(data.num_nodes, data.edge_index), a.split_seed,
                                 a.val_ratio, a.test_ratio)
    torch.save(dict(edge_index=visible.to_edge_index(), splits=splits, split_seed=a.split_seed), run / "split.pt")
    x = data.x.to(a.device)
    encoder = make_encoder(a, x.shape[1], seed).to(a.device)
    head = PairHead(encoder.hidden_dim).to(a.device)
    optimizer = torch.optim.AdamW(list(encoder.parameters())+list(head.parameters()),
                                  lr=a.learning_rate, weight_decay=a.weight_decay)
    train_pairs, train_labels = splits["train"]
    # Reuse the disjoint training negative pool, sampled against the FULL graph.
    train_negatives = train_pairs[train_labels == 0].tolist()
    training_rng = random.Random(seed)
    splits = {s: (pairs.to(a.device), labels.to(a.device)) for s, (pairs, labels) in splits.items()}
    write_json(run / "config.json", dict(vars(a), seed=seed))
    best, best_epoch = -1., 0
    fields = ("dataset", "mode", "seed", "epoch", "train_loss", "train_supervision_edges",
              "actual_train_supervision_ratio", "validation_auc", "validation_ap", "best_epoch")
    with CSVLog(run / "epochs.csv", fields) as log:
        for epoch in range(1, a.epochs+1):
            message_program, pairs, labels = sample_training_edges(
                visible, train_negatives, a.train_supervision_ratio, training_rng)
            supervision_count = len(labels) // 2
            pairs, labels = pairs.to(a.device), labels.to(a.device)
            encoder.train(); head.train()
            optimizer.zero_grad(set_to_none=True)
            nodes = encoder(message_program, x).node_embeddings
            loss = nn.functional.binary_cross_entropy_with_logits(head(nodes, pairs), labels)
            loss.backward(); optimizer.step()
            encoder.eval(); head.eval()
            with torch.no_grad():
                nodes = encoder(visible, x).node_embeddings
                pairs, labels = splits["val"]
                val = binary_metrics(labels, head(nodes, pairs))
            if val["auc"] > best:
                best, best_epoch = val["auc"], epoch
                save_downstream(run / "best.pt", encoder, head, a, seed, epoch, best,
                                 validation_ap=val["ap"], split_seed=a.split_seed)
            log.write(dict(dataset=a.dataset, mode=a.mode, seed=seed, epoch=epoch, train_loss=loss.item(),
                           train_supervision_edges=supervision_count,
                           actual_train_supervision_ratio=supervision_count/visible.num_edges,
                           validation_auc=val["auc"], validation_ap=val["ap"], best_epoch=best_epoch))
    ckpt = torch.load(run / "best.pt", map_location=a.device, weights_only=True)
    encoder.load_state_dict(ckpt["encoder_state_dict"]); head.load_state_dict(ckpt["head_state_dict"])
    encoder.eval(); head.eval()
    with torch.no_grad():
        nodes = encoder(visible, x).node_embeddings
        pairs, labels = splits["test"]
        test = binary_metrics(labels, head(nodes, pairs))
    return dict(dataset=a.dataset, mode=a.mode, seed=seed, best_epoch=best_epoch,
                validation_auc=best, validation_ap=ckpt["validation_ap"],
                test_auc=test["auc"], test_ap=test["ap"])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p); add_training_args(p, "link"); add_transfer_args(p)
    p.add_argument("--dataset", choices=("Cora", "Citeseer"), default="Cora")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=.1)
    p.add_argument("--test_ratio", type=float, default=.1)
    p.add_argument("--train_supervision_ratio", type=float, default=.15,
                   help="fraction of training positive edges hidden for supervision each epoch; at least one")
    a = p.parse_args(argv)
    if not 0 < a.train_supervision_ratio <= 1:
        p.error("train_supervision_ratio must be in (0,1]")
    root = new_run(a.output_dir)
    results = [run_seed(a, s, new_run(root / f"seed_{s}")) for s in (a.seeds or [a.seed])]
    summarize(root, results, ("validation_auc", "validation_ap", "test_auc", "test_ap"))


if __name__ == "__main__":
    main()
