"""Independent MLP/GCN/GAT and Graph-IR comparisons on public Planetoid masks."""

import argparse
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

if not __package__:
    # PyCharm Run executes this file directly, without a package context.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "Graph"

from .ir import compile_graph
from .finetune_node import run_seed as run_graph_ir
from .utils import (add_model_args, add_training_args, add_transfer_args, seed_all, new_run,
                    CSVLog, write_json, load_planetoid, rewire_graph, accuracy, summarize)


class NodeBaseline(nn.Module):
    def __init__(self, kind, input_dim, hidden_dim, classes, layers, heads, dropout):
        super().__init__()
        if layers < 1:
            raise ValueError("baseline_layers must be positive")
        self.kind, self.dropout = kind, dropout
        dims = [input_dim] + [hidden_dim]*(layers-1) + [classes]
        if kind == "mlp":
            self.layers = nn.ModuleList([nn.Linear(i, o) for i, o in zip(dims, dims[1:])])
        else:
            from torch_geometric.nn import GCNConv, GATConv
            self.layers = nn.ModuleList([
                GCNConv(i, o) if kind == "gcn" else GATConv(i, o, heads=heads, concat=False, dropout=dropout)
                for i, o in zip(dims, dims[1:])])

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x) if self.kind == "mlp" else layer(x, edge_index)
            if i < len(self.layers)-1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


def run_baseline(a, seed, run):
    data, classes = load_planetoid(a.data_dir, a.dataset)
    program = compile_graph(data.num_nodes, data.edge_index)
    stats = dict(structure="original", successful_swaps=0, changed_edge_ratio=0.)
    if a.structure == "rewired":
        program, stats = rewire_graph(program, a.rewire_seed, a.swap_multiplier)
    x, y, edge_index = data.x.to(a.device), data.y.to(a.device), program.to_edge_index(a.device)
    masks = {s: getattr(data, f"{s}_mask").to(a.device) for s in ("train", "val", "test")}
    seed_all(seed)
    model = NodeBaseline(a.model, x.shape[1], a.hidden_dim, classes, a.baseline_layers,
                         a.num_heads, a.dropout).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    write_json(run / "config.json", dict(vars(a), seed=seed, structure_stats=stats))
    best, best_epoch = -1., 0
    with CSVLog(run / "epochs.csv", ("dataset", "model", "structure", "seed", "epoch", "train_loss",
                 "train_accuracy", "validation_accuracy", "best_epoch")) as log:
        for epoch in range(1, a.epochs+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index)
            loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
            loss.backward(); optimizer.step(); model.eval()
            with torch.no_grad():
                logits = model(x, edge_index)
                val = accuracy(logits[masks["val"]], y[masks["val"]])
                train = accuracy(logits[masks["train"]], y[masks["train"]])
            if val > best:
                best, best_epoch = val, epoch
                torch.save(dict(model_state_dict=model.state_dict(), config=dict(vars(a), data_dir=str(a.data_dir),
                    output_dir=str(a.output_dir), checkpoint=str(a.checkpoint) if a.checkpoint else None),
                    seed=seed, epoch=epoch, validation_accuracy=val,
                    split_masks={k: v.cpu() for k, v in masks.items()}, edge_index=edge_index.cpu()), run / "best.pt")
            log.write(dict(dataset=a.dataset, model=a.model, structure=a.structure, seed=seed,
                           epoch=epoch, train_loss=loss.item(), train_accuracy=train,
                           validation_accuracy=val, best_epoch=best_epoch))
    ckpt = torch.load(run / "best.pt", map_location=a.device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        test = accuracy(logits[masks["test"]], y[masks["test"]])
    return dict(dataset=a.dataset, model=a.model, structure=a.structure, seed=seed, best_epoch=best_epoch,
                validation_accuracy=best, test_accuracy=test, structure_stats=stats)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_model_args(p); add_training_args(p, "baselines"); add_transfer_args(p)
    p.add_argument("--model", choices=("mlp", "gcn", "gat", "scratch", "pretrained", "all"), default="mlp")
    p.add_argument("--dataset", choices=("Cora", "Citeseer"), default="Cora")
    p.add_argument("--baseline_layers", type=int, default=2)
    p.add_argument("--structure", choices=("original", "rewired"), default="original")
    p.add_argument("--rewire_seed", type=int, default=42)
    p.add_argument("--swap_multiplier", type=float, default=5.)
    a = p.parse_args(argv)
    if a.model in ("pretrained", "all") and a.checkpoint is None:
        p.error("pretrained/all requires --checkpoint")
    root = new_run(a.output_dir)
    kinds = ("mlp", "gcn", "gat", "scratch", "pretrained") if a.model == "all" else (a.model,)
    for kind in kinds:
        a.model = kind
        a.mode = kind if kind in ("scratch", "pretrained") else "scratch"
        folder = new_run(root / kind)
        runner = run_graph_ir if kind in ("scratch", "pretrained") else run_baseline
        results = [runner(a, s, new_run(folder / f"seed_{s}")) for s in (a.seeds or [a.seed])]
        summarize(folder, results, ("validation_accuracy", "test_accuracy"))


if __name__ == "__main__":
    main()
