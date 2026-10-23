"""Ordinary graph storage, reproducible subsets, and complete-edge masking."""

from dataclasses import dataclass
from pathlib import Path
import random
import torch
from torch import Tensor
from .ir import GraphIRProgram, compile_graph

FAMILIES = ("er", "ba", "ws", "sbm")
SPLITS = ("train", "val", "test", "ood")
DATA_ROOT = Path(__file__).resolve().parent / "datasets"


@dataclass
class GraphData:
    num_nodes: int
    edge_index: Tensor


def load_graph(path):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return GraphData(int(raw["num_nodes"]), raw["edge_index"])


def pairs_index(pairs):
    return torch.tensor(list(pairs), dtype=torch.long).reshape(-1, 2).t().contiguous()


def split_paths(root, split, limit=None, seed=0, families=None):
    families = FAMILIES if families is None else tuple(families)
    if not families or len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError("families must be a nonempty selection of distinct supported graph families")
    groups = [sorted((Path(root) / split / family).glob("*.pt")) for family in families]
    rng = random.Random(seed)
    for group in groups:
        rng.shuffle(group)
    if limit is None:
        return [p for group in groups for p in group]
    if limit < 1:
        raise ValueError("subset size must be positive")
    # Round-robin handles unequal family sizes without losing the budget.
    selected = []
    for i in range(max(map(len, groups), default=0)):
        for group in groups:
            if i < len(group):
                selected.append(group[i])
                if len(selected) == limit:
                    return selected
    return selected


def sample_non_edges(num_nodes, full_edges, count, rng, excluded=()):
    """Unique negatives, excluding the complete original edge set."""
    forbidden = set(full_edges) | set(excluded)
    available = num_nodes * (num_nodes - 1) // 2 - len(forbidden)
    if count > available:
        raise ValueError(f"requested {count} negatives but only {available} nonedges exist")
    # Enumerate dense small complements; rejection sampling stays cheap on sparse citation graphs.
    if available < 4 * count:
        candidates = [(u, v) for u in range(num_nodes) for v in range(u + 1, num_nodes)
                      if (u, v) not in forbidden]
        return rng.sample(candidates, count)
    chosen = set()
    while len(chosen) < count:
        u, v = rng.sample(range(num_nodes), 2)
        pair = (min(u, v), max(u, v))
        if pair not in forbidden:
            chosen.add(pair)
    return sorted(chosen)


@dataclass
class MaskedEdgeBatch:
    program: GraphIRProgram
    positive_pairs: Tensor
    negative_pairs: Tensor
    original_edges: int

    @property
    def actual_mask_ratio(self):
        return len(self.positive_pairs) / max(self.original_edges, 1)


def sample_masked_edges(graph, mask_ratio, rng):
    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must be in (0,1]")
    original = compile_graph(graph.num_nodes, graph.edge_index)
    edges = [e.endpoints for e in original.instructions]
    degree = [0] * graph.num_nodes
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    capacity = graph.num_nodes * (graph.num_nodes - 1) // 2 - len(edges)
    target = min(max(1, round(len(edges) * mask_ratio)), capacity)
    order = list(edges)
    rng.shuffle(order)
    hidden = set()
    for u, v in order:
        if len(hidden) >= target:
            break
        if degree[u] > 1 and degree[v] > 1:
            hidden.add((u, v))
            degree[u] -= 1
            degree[v] -= 1
    if not hidden:
        return None
    negatives = sample_non_edges(graph.num_nodes, edges, len(hidden), rng)
    # Only visible edges enter compilation; neither full degree nor hidden edge IDs survive.
    visible = compile_graph(graph.num_nodes, pairs_index(e for e in edges if e not in hidden))
    return MaskedEdgeBatch(visible, pairs_index(sorted(hidden)).t(),
                           pairs_index(negatives).t(), len(edges))
