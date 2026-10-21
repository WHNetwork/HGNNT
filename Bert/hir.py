"""Lossless DIRECT-only H-IR; symbols are identities, never numeric features."""

from dataclasses import dataclass, replace

import torch
from torch import Tensor


@dataclass(frozen=True)
class EdgeInstruction:
    """One complete ORDER / EDGE / DIRECT / END_EDGE instruction.

    edge_symbol indexes the original edge, even after sorting by visible order.
    members are node symbols indexing rows in the program's symbol table.
    """

    edge_symbol: int
    members: tuple[int, ...]

    @property
    def order(self) -> int:
        return len(self.members)

    @property
    def operations(self) -> tuple:
        return (("ORDER", (self.order,)), ("EDGE", (self.edge_symbol,)),
                ("DIRECT", self.members), ("END_EDGE", ()))


@dataclass(frozen=True)
class HIRProgram:
    instructions: tuple[EdgeInstruction, ...]
    node_symbols: tuple[int, ...]
    x: Tensor
    num_nodes: int
    num_hyperedges: int

    def to(self, device) -> "HIRProgram":
        return replace(self, x=self.x.to(device))

    def to_hyperedge_index(self) -> Tensor:
        """Recover membership in original edge/feature-row coordinates."""
        pairs = [(n, ir.edge_symbol) for ir in self.instructions for n in ir.members]
        return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()


def compile_hypergraph(num_nodes: int, hyperedge_index: Tensor, x: Tensor | None = None,
                       num_hyperedges: int | None = None) -> HIRProgram:
    """Compile [2, K] incidences; optional x is retained for data compatibility.

    Symbols use dense source coordinates: node n maps to x[n], edge e to output
    row e. Repeated incidences are one membership; duplicate edges stay distinct.
    With no explicit edge count, max(edge_id)+1 is used (zero for no incidences).
    Supply num_hyperedges to preserve trailing empty edges or all-empty graphs.
    Isolated nodes are always retained in the symbol table. The structure-only
    Encoder does not read x, including when nonempty features are supplied.
    """
    if x is None:
        x = torch.empty((num_nodes, 0), dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] != num_nodes or num_nodes < 0:
        raise ValueError("x must have shape [num_nodes, feature_dim]")
    if hyperedge_index.ndim != 2 or hyperedge_index.shape[0] != 2:
        raise ValueError("hyperedge_index must have shape [2, num_incidences]")
    if hyperedge_index.dtype != torch.long:
        raise ValueError("hyperedge_index must be a torch.long tensor")
    nodes, edges = hyperedge_index.detach().cpu().tolist()
    inferred = max(edges, default=-1) + 1
    count = inferred if num_hyperedges is None else num_hyperedges
    if count < inferred or count < 0 or any(e < 0 for e in edges):
        raise ValueError("edge symbols must be in [0, num_hyperedges)")
    if any(n < 0 or n >= num_nodes for n in nodes):
        raise ValueError("node symbols must index the node feature table")
    members = [set() for _ in range(count)]
    for n, e in zip(nodes, edges):
        members[e].add(n)
    instructions = [EdgeInstruction(e, tuple(sorted(group)))
                    for e, group in enumerate(members)]
    # Stable ties are a serialization choice, with no positional model signal.
    instructions.sort(key=lambda ir: ir.order)
    return HIRProgram(tuple(instructions), tuple(range(num_nodes)), x,
                      num_nodes, count)
