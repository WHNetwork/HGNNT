"""Lossless compiler for undirected simple graphs; IDs only bind symbols."""

from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class EdgeInstruction:
    edge_symbol: int
    endpoints: tuple[int, int]

    @property
    def operations(self):
        return (("EDGE", (self.edge_symbol,)), ("CONNECT", self.endpoints), ("END_EDGE", ()))


@dataclass(frozen=True)
class GraphIRProgram:
    num_nodes: int
    node_symbols: tuple[int, ...]
    instructions: tuple[EdgeInstruction, ...]

    def __post_init__(self):
        if self.num_nodes < 0 or self.node_symbols != tuple(range(self.num_nodes)):
            raise ValueError("node_symbols must preserve all source node rows")
        seen = set()
        for row, edge in enumerate(self.instructions):
            u, v = edge.endpoints
            pair = tuple(sorted((u, v)))
            if edge.edge_symbol != row or not (0 <= u < self.num_nodes and 0 <= v < self.num_nodes):
                raise ValueError("invalid edge symbol or endpoint")
            if u == v or pair in seen:
                raise ValueError("simple graphs require distinct endpoints and unique edges")
            seen.add(pair)

    @property
    def num_edges(self):
        return len(self.instructions)

    def to_edge_index(self, device=None):
        return to_edge_index(self, device)


def compile_graph(num_nodes: int, edge_index: Tensor) -> GraphIRProgram:
    if not isinstance(num_nodes, int) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be LongTensor[2, K]")
    pairs = set()
    for u, v in edge_index.detach().cpu().t().tolist():
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("endpoint outside node symbol table")
        if u == v:
            raise ValueError("self loops are not supported")
        pairs.add(tuple(sorted((u, v))))
    return GraphIRProgram(num_nodes, tuple(range(num_nodes)), tuple(
        EdgeInstruction(i, pair) for i, pair in enumerate(sorted(pairs))))


def to_edge_index(program: GraphIRProgram, device=None) -> Tensor:
    pairs = [pair for edge in program.instructions
             for pair in (edge.endpoints, edge.endpoints[::-1])]
    return torch.tensor(pairs, dtype=torch.long, device=device).reshape(-1, 2).t().contiguous()
