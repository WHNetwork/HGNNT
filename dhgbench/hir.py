"""Deterministic H-IR compiler, parser and bounded BERT input chunks.

Repeated incidences are one membership; equal edges with different IDs remain
separate edges. No features or random numbers are used by this module.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Integral


PAD, EDGE, END_EDGE, DIRECT, EXTEND, PATCH, REF, DROP, ADD, FRAGMENT = range(10)
NODE, EDGE_ID, ORDER = range(10, 13)  # embedding types for numeric payloads
NUM_TYPES = 13
MODE_NAMES = {DIRECT: "DIRECT", EXTEND: "EXTEND", PATCH: "PATCH"}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class TokenSpace:
    num_nodes: int
    num_edges: int

    def node(self, node):
        return NUM_TYPES + node

    def edge(self, edge):
        return NUM_TYPES + self.num_nodes + edge

    def order(self, order):
        return NUM_TYPES + self.num_nodes + self.num_edges + order


@dataclass(frozen=True)
class EdgeIR:
    edge_id: int
    order: int
    mode: int
    ref: int = -1
    drop: tuple = ()
    add: tuple = ()

    def tokens(self, space):
        head = (EDGE, space.edge(self.edge_id), space.order(self.order), self.mode)
        if self.mode in (DIRECT, FRAGMENT):
            body = tuple(space.node(v) for v in self.add)
        else:
            body = (REF, space.edge(self.ref))
            if self.mode == PATCH:
                body += (DROP,) + tuple(space.node(v) for v in self.drop)
            body += (ADD,) + tuple(space.node(v) for v in self.add)
        return head + body + (END_EDGE,)


@dataclass(frozen=True)
class ParsedHIR:
    num_nodes: int
    hyperedges: tuple
    instructions: tuple

    def to_hyperedge_index(self):
        import torch
        pairs = [(v, e) for e, edge in enumerate(self.hyperedges) for v in edge]
        return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).t().contiguous()

    def incidence_matrix(self):
        import torch
        index = self.to_hyperedge_index()
        return torch.sparse_coo_tensor(
            index, torch.ones(index.shape[1]),
            (self.num_nodes, len(self.hyperedges)),
        ).coalesce()


@dataclass(frozen=True)
class HIRProgram(ParsedHIR):
    tokens: tuple
    source_edge_ids: tuple
    duplicate_incidences: int

    @property
    def space(self):
        return TokenSpace(self.num_nodes, len(self.hyperedges))

    @property
    def stats(self):
        modes = Counter(MODE_NAMES[ir.mode] for ir in self.instructions)
        count = len(self.hyperedges)
        return {
            "num_nodes": self.num_nodes, "num_hyperedges": count,
            "token_count": len(self.tokens),
            "reference_ratio": (count - modes["DIRECT"]) / max(count, 1),
            "mode_counts": {name: modes[name] for name in MODE_NAMES.values()},
            "order_counts": dict(sorted(Counter(map(len, self.hyperedges)).items())),
            "duplicate_incidences_removed": self.duplicate_incidences,
            "duplicate_hyperedges_preserved": count - len(set(self.hyperedges)),
        }


def compile_hypergraph(num_nodes, hyperedge_index, num_hyperedges=None):
    """Compile [2, incidences] with arbitrary nonnegative source edge IDs.

    If num_hyperedges is supplied, IDs must be in [0, num_hyperedges); absent
    columns are explicit empty edges. Otherwise only represented IDs are edges.
    Canonical edge IDs follow (cardinality, lexicographic members). Source IDs
    break ties between identical edges, whose emitted definitions are identical
    under any permutation of the input columns/IDs.
    """
    num_nodes = _integer(num_nodes, "num_nodes")
    rows = hyperedge_index.tolist() if hasattr(hyperedge_index, "tolist") else hyperedge_index
    if len(rows) != 2 or len(rows[0]) != len(rows[1]):
        raise ValueError("hyperedge_index must have shape [2, num_incidences]")
    groups = {}
    if num_hyperedges is not None:
        num_hyperedges = _integer(num_hyperedges, "num_hyperedges")
        groups = {e: set() for e in range(num_hyperedges)}
    for node, edge in zip(*rows):
        node, edge = _integer(node, "node ID"), _integer(edge, "edge ID")
        if node >= num_nodes:
            raise ValueError(f"Node ID {node} outside [0, {num_nodes})")
        if num_hyperedges is not None and edge >= num_hyperedges:
            raise ValueError("Edge ID outside declared num_hyperedges")
        groups.setdefault(edge, set()).add(node)
    canonical = sorted(((tuple(sorted(v)), e) for e, v in groups.items()),
                       key=lambda pair: (len(pair[0]), pair[0], pair[1]))
    edges = tuple(pair[0] for pair in canonical)
    space = TokenSpace(num_nodes, len(edges))
    previous_by_node = defaultdict(list)
    edge_sets, instructions = [], []
    for i, members in enumerate(edges):
        target = frozenset(members)
        # Token costs: DIRECT=5+k, EXTEND=8+adds, PATCH=9+drops+adds.
        # Ties prefer DIRECT, then EXTEND, then PATCH, then smallest ref ID.
        best = (5 + len(members), 0, -1)
        mode, reference = DIRECT, -1
        overlaps = Counter()
        for v in members:
            overlaps.update(previous_by_node[v])
        for ref, shared in sorted(overlaps.items()):
            base_size = len(edges[ref])
            if shared == base_size and base_size < len(members):
                candidate, kind = (8 + len(members) - base_size, 1, ref), EXTEND
            else:
                candidate = (9 + base_size + len(members) - 2 * shared, 2, ref)
                kind = PATCH
            if candidate < best:
                best, mode, reference = candidate, kind, ref
        if reference < 0:
            ir = EdgeIR(i, len(members), DIRECT, add=members)
        else:
            base = edge_sets[reference]
            ir = EdgeIR(i, len(members), mode, reference,
                        tuple(v for v in edges[reference] if v not in target),
                        tuple(v for v in members if v not in base))
        instructions.append(ir)
        edge_sets.append(target)
        for v in members:
            previous_by_node[v].append(i)
    tokens = tuple(t for ir in instructions for t in ir.tokens(space))
    return HIRProgram(num_nodes, edges, tuple(instructions), tokens,
                      tuple(pair[1] for pair in canonical),
                      len(rows[0]) - sum(map(len, edges)))


def parse_hir(tokens, num_nodes, num_hyperedges):
    """Strict parser for complete logical H-IR (not BERT-only fragments)."""
    num_nodes = _integer(num_nodes, "num_nodes")
    num_hyperedges = _integer(num_hyperedges, "num_hyperedges")
    space = TokenSpace(num_nodes, num_hyperedges)
    stream = iter(tokens)
    edges, instructions = [], []

    def take():
        try:
            return _integer(next(stream), "token")
        except StopIteration:
            raise ValueError("Incomplete H-IR syntax") from None

    def expect(symbol):
        if take() != symbol:
            raise ValueError(f"Expected H-IR symbol {symbol}")

    def edge_id():
        value = take() - space.edge(0)
        if not 0 <= value < num_hyperedges:
            raise ValueError("Invalid edge/reference ID")
        return value

    def nodes_until(stop):
        members = []
        while True:
            token = take()
            if token == stop:
                break
            node = token - space.node(0)
            if not 0 <= node < num_nodes:
                raise ValueError("Invalid node ID or unexpected H-IR symbol")
            members.append(node)
        if members != sorted(set(members)):
            raise ValueError("Node lists must be strictly increasing")
        return tuple(members)

    for i in range(num_hyperedges):
        expect(EDGE)
        if edge_id() != i:
            raise ValueError("Edge definitions must use consecutive canonical IDs")
        order = take() - space.order(0)
        if not 0 <= order <= num_nodes:
            raise ValueError("Invalid hyperedge order")
        mode, reference, drops = take(), -1, ()
        if mode == DIRECT:
            adds = nodes_until(END_EDGE)
            recovered = frozenset(adds)
        elif mode in (EXTEND, PATCH):
            expect(REF)
            reference = edge_id()
            if reference >= i:
                raise ValueError("Reference must name an already defined edge")
            base = frozenset(edges[reference])
            if mode == PATCH:
                expect(DROP)
                drops = nodes_until(ADD)
            else:
                expect(ADD)
            adds = nodes_until(END_EDGE)
            if not set(drops) <= base or set(adds) & base:
                raise ValueError("Illegal DROP/ADD membership")
            recovered = (base - set(drops)) | set(adds)
            if mode == EXTEND and (len(base) >= order or not adds):
                raise ValueError("EXTEND requires a strict lower-order subset")
            if mode == PATCH and not (base & recovered):
                raise ValueError("PATCH requires overlapping edges")
        else:
            raise ValueError("Unknown H-IR expression mode")
        if len(recovered) != order:
            raise ValueError("Declared order differs from recovered cardinality")
        edges.append(tuple(sorted(recovered)))
        instructions.append(EdgeIR(i, order, mode, reference, drops, adds))
    end = object()
    if next(stream, end) is not end:
        raise ValueError("Unexpected tokens after final edge")
    return ParsedHIR(num_nodes, tuple(edges), tuple(instructions))


@dataclass(frozen=True)
class HIRChunk:
    tokens: tuple
    orders: tuple
    # (EDGE position, canonical edge ID, contribution weight)
    readouts: tuple


def chunk_hir(program, max_seq_length):
    """Lower logical H-IR into self-contained bounded BERT blocks.

    Copy missing referenced definitions as DIRECT context. If context+target
    cannot fit, materialize the target as DIRECT. Split oversized DIRECT edges
    into member fragments; their EDGE states are member-count weighted later.
    Context copies never create additional output hyperedges.
    """
    if not isinstance(max_seq_length, Integral) or max_seq_length < 6:
        raise ValueError("max_seq_length must be at least 6")
    space, chunks = program.space, []
    tokens, orders, readouts, defined = [], [], [], set()
    contexts = fallbacks = fragments = 0

    def flush():
        if tokens:
            chunks.append(HIRChunk(tuple(tokens), tuple(orders), tuple(readouts)))
        tokens.clear()
        orders.clear()
        readouts.clear()
        defined.clear()

    def append(ir, output=False, weight=1.0):
        record = ir.tokens(space)
        if output:
            readouts.append((len(tokens), ir.edge_id, weight))
        tokens.extend(record)
        orders.extend([ir.order] * len(record))
        defined.add(ir.edge_id)

    def direct(edge_id):
        members = program.hyperedges[edge_id]
        return EdgeIR(edge_id, len(members), DIRECT, add=members)

    for original in program.instructions:
        ir = original
        length = len(ir.tokens(space))
        if ir.ref >= 0:
            context = direct(ir.ref)
            extra = 0 if ir.ref in defined else len(context.tokens(space))
            if len(tokens) + extra + length > max_seq_length:
                flush()
                extra = len(context.tokens(space))
            if extra + length > max_seq_length:
                ir = direct(ir.edge_id)
                length = len(ir.tokens(space))
                fallbacks += 1
            elif ir.ref not in defined:
                append(context)
                contexts += 1
        if length > max_seq_length:
            flush()
            members = program.hyperedges[ir.edge_id]
            capacity = max_seq_length - 5
            for start in range(0, len(members), capacity):
                part = members[start:start + capacity]
                append(EdgeIR(ir.edge_id, ir.order, FRAGMENT, add=part),
                       output=True, weight=len(part) / len(members))
                fragments += 1
                flush()
        else:
            if len(tokens) + length > max_seq_length:
                flush()
            append(ir, output=True)
    flush()
    stats = {
        "num_chunks": len(chunks),
        "max_chunk_length": max((len(c.tokens) for c in chunks), default=0),
        "bert_token_count": sum(len(c.tokens) for c in chunks),
        "context_definitions": contexts, "direct_fallbacks": fallbacks,
        "long_edge_fragments": fragments,
    }
    return tuple(chunks), stats
