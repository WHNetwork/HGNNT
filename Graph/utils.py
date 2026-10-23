"""Shared experiment I/O, transfer, symmetric scoring, and ordinary rewiring."""

import argparse
import csv
import json
from pathlib import Path
import random
import statistics
import numpy as np
import torch
from torch import nn
from .data import DATA_ROOT, pairs_index
from .encoder import ARCHITECTURE_VERSION, GraphIREncoder
from .ir import compile_graph

RUN_ROOT = Path(__file__).resolve().parent / "runs"
CORE_KEYS = ("hidden_dim", "num_blocks", "num_heads", "dropout", "ffn_expansion")


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def add_model_args(parser):
    parser.add_argument("--hidden_dim", type=positive_int, default=256)
    parser.add_argument("--num_blocks", type=positive_int, default=4)
    parser.add_argument("--num_heads", type=positive_int, default=4)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--ffn_expansion", type=positive_int, default=4)
    parser.add_argument("--max_tokens", type=positive_int,
                        help="optional limit on IR tokens + node slots + GRAPH; never truncates")
    parser.add_argument("--max_relations", type=positive_int,
                        help="optional limit on sparse attention relations; never truncates")


def add_training_args(parser, name, epochs=200):
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=positive_int, default=epochs)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--output_dir", type=Path, default=RUN_ROOT / name)


def add_transfer_args(parser):
    parser.add_argument("--mode", choices=("scratch", "pretrained"), default="scratch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data_dir", type=Path, default=DATA_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+")


def model_config(args):
    return {k: getattr(args, k) for k in (*CORE_KEYS, "max_tokens", "max_relations")}


def seed_all(seed):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def new_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(_plain(data), stream, indent=2)


class CSVLog:
    def __init__(self, path, fields):
        self.stream = Path(path).open("x", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)
        self.stream.flush()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


def binary_metrics(labels, logits):
    from sklearn.metrics import roc_auc_score, average_precision_score
    labels = labels.detach().cpu().numpy()
    # Sigmoid is monotone; logits avoid saturated probability ties.
    scores = logits.detach().cpu().numpy()
    return dict(auc=float(roc_auc_score(labels, scores)), ap=float(average_precision_score(labels, scores)))


def accuracy(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


class PairHead(nn.Module):
    """An undirected scorer invariant to exchanging the two endpoint rows."""
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, nodes, pairs):
        u, v = nodes[pairs[:, 0]], nodes[pairs[:, 1]]
        return self.mlp(torch.cat((u+v, (u-v).abs(), u*v), -1)).squeeze(-1)


def checkpoint_encoder_config(checkpoint):
    config = checkpoint.get("encoder_config", {})
    version = config.get("architecture_version")
    if version != ARCHITECTURE_VERSION:
        raise ValueError(f"Incompatible encoder checkpoint architecture {version!r}; "
                         f"expected {ARCHITECTURE_VERSION!r}. Old five-stage Encoder weights "
                         "are unsupported; pretrain the new architecture again.")
    return config


def load_encoder_checkpoint(encoder, checkpoint):
    checkpoint_encoder_config(checkpoint)
    encoder.load_state_dict(checkpoint["encoder_state_dict"], strict=True)


def make_encoder(args, feature_dim, seed):
    """Strict core loading FIRST, then initialize a fresh dataset feature adapter.

    Scratch and transfer use identical CLI architecture, public splits, and head/
    adapter RNG. Token/relation limits are runtime budgets, independent of weights.
    """
    seed_all(seed)
    encoder = GraphIREncoder(**model_config(args))
    if args.mode == "pretrained":
        if args.checkpoint is None:
            raise ValueError("pretrained mode requires --checkpoint")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        config = checkpoint_encoder_config(ckpt)
        mismatches = {k: (config[k], getattr(args, k)) for k in CORE_KEYS if config[k] != getattr(args, k)}
        if mismatches or config.get("feature_dim") is not None:
            raise ValueError(f"expected structure-only checkpoint with matching CLI architecture: {mismatches}")
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
    # Reset after core construction/loading so both modes initialize new modules identically.
    seed_all(seed+1)
    if feature_dim is not None:
        encoder.add_feature_adapter(feature_dim)
    return encoder


def save_downstream(path, encoder, head, args, seed, epoch, metric, **extra):
    torch.save(dict(encoder_config=encoder.config, encoder_state_dict=encoder.state_dict(),
                    head_state_dict=head.state_dict(), epoch=epoch, seed=seed,
                    validation_metric=metric, experiment_config=_plain(vars(args)), **extra), path)


def load_planetoid(root, name):
    from torch_geometric.datasets import Planetoid
    dataset = Planetoid(str(Path(root) / "Planetoid"), name=name, split="public")
    return dataset[0], dataset.num_classes


def rewire_graph(program, seed=42, multiplier=5.):
    """Reproducible simple-graph double-edge swaps, including partial success.

    All four selected nodes are distinct. Each accepted swap preserves every
    node's degree exactly; failed proposals never change the graph.
    """
    if multiplier < 0:
        raise ValueError("swap_multiplier must be nonnegative")
    rng = random.Random(seed)
    edges = [e.endpoints for e in program.instructions]
    original, current = set(edges), set(edges)
    target = round(multiplier*len(edges))
    attempts, success = 0, 0
    max_attempts = max(100, 20*target)
    while success < target and attempts < max_attempts and len(edges) >= 2:
        attempts += 1
        i, j = rng.sample(range(len(edges)), 2)
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        if rng.random() < .5:
            c, d = d, c
        first, second = tuple(sorted((a, c))), tuple(sorted((b, d)))
        if first in current or second in current:
            continue
        current.remove(edges[i]); current.remove(edges[j])
        current.update((first, second))
        edges[i], edges[j] = first, second
        success += 1
    result = compile_graph(program.num_nodes, pairs_index(edges))
    return result, dict(structure="rewired", target_swaps=target, successful_swaps=success,
        attempts=attempts, max_attempts=max_attempts, changed_edge_ratio=len(original-current)/max(len(original), 1),
        stop_reason="target_reached" if success == target else "proposal_budget_exhausted")


def summarize(root, results, metrics):
    fields = tuple(dict.fromkeys(k for result in results for k in result))
    with CSVLog(root / "results.csv", fields) as log:
        for result in results:
            log.write(result)
    with CSVLog(root / "summary.csv", ("metric", "mean", "std", "num_seeds")) as log:
        for metric in metrics:
            values = [r[metric] for r in results]
            log.write(dict(metric=metric, mean=statistics.mean(values),
                           std=statistics.stdev(values) if len(values)>1 else 0., num_seeds=len(values)))
    print(results)
