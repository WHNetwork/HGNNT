"""Standalone tensor data, with no DHG-Bench dependency.

Each .pt is a dictionary: num_nodes, hyperedge_index (long [2,K]), optional x.
IDs index symbol/output rows. Optional num_hyperedges preserves trailing empty
edges; otherwise max(edge_id)+1 is inferred, including intermediate empty edges.
Structure-only loading ignores external features and returns an empty [N,0] x.
Existing train/val/test/ood splits and their four families are never repartitioned.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

FAMILIES = ("random", "chung_lu", "dcsbm", "watts_strogatz")
SPLITS = ("train", "val", "test", "ood")


@dataclass
class Hypergraph:
    num_nodes: int
    num_hyperedges: int
    hyperedge_index: Tensor
    x: Tensor


def load_graph(path: str | Path) -> Hypergraph:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    num_nodes = int(raw["num_nodes"])
    # No full-graph compilation or statistics are passed to the Encoder.
    index = raw["hyperedge_index"].long()
    inferred = int(index[1].max()) + 1 if index.shape[1] else 0
    return Hypergraph(num_nodes, int(raw.get("num_hyperedges", inferred)), index,
                      torch.empty((num_nodes, 0), dtype=torch.float32))
