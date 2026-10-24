"""Deterministic HIR sequence adapter; local IDs are references, not distances."""

from dataclasses import dataclass
from enum import IntEnum

from .ir import EdgeInstruction, GraphIRProgram
from .tokenizer import GraphTokenizer, TokenType

TOKENIZER_VERSION = "hir_sequence_deterministic_v1"
MAX_NODES = 65536


class SequenceType(IntEnum):
    EDGE = 0
    CONNECT = 1
    NODE_REF = 2
    END_EDGE = 3
    ISOLATE = 4
    GRAPH = 5
    PAD = 6


def hex_digits(node_id):
    """Four most-significant-first digits, with no truncation or wraparound."""
    if not isinstance(node_id, int) or not 0 <= node_id < MAX_NODES:
        raise ValueError("node ID must be an integer in [0, 65535]")
    return tuple((node_id >> shift) & 15 for shift in (12, 8, 4, 0))


def digits_to_id(digits):
    if len(digits) != 4 or any(not isinstance(d, int) or not 0 <= d < 16 for d in digits):
        raise ValueError("expected four integer hexadecimal digits in [0, 15]")
    return sum(d << shift for d, shift in zip(digits, (12, 8, 4, 0)))


@dataclass(frozen=True)
class SequenceProgram:
    token_types: tuple[int, ...]
    # -1 means this position has no associated node identity.
    node_ids: tuple[int, ...]
    node_symbols: tuple[int, ...]
    node_positions: tuple[tuple[int, ...], ...]
    instruction_spans: tuple[tuple[int, int], ...]
    edge_positions: tuple[int, ...]
    edge_symbols: tuple[int, ...]
    edge_endpoints: tuple[tuple[int, int], ...]
    isolated_nodes: tuple[int, ...]
    graph_position: int

    @property
    def token_length(self):
        return len(self.token_types)


class HIRSequenceTokenizer:
    def __init__(self, max_length=8192, tokenizer_version=TOKENIZER_VERSION):
        if tokenizer_version != TOKENIZER_VERSION:
            raise ValueError(f"unsupported sequence tokenizer: {tokenizer_version!r}")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self.config = dict(max_length=max_length, tokenizer_version=tokenizer_version)
        self.ir_tokenizer = GraphTokenizer()

    def __call__(self, program: GraphIRProgram) -> SequenceProgram:
        if program.num_nodes > MAX_NODES:
            raise ValueError(f"{program.num_nodes} nodes exceed the 65536 local node ID capacity")
        tok = self.ir_tokenizer(program)
        isolates = tuple(u for u, refs in enumerate(tok.symbol_occurrences) if not refs)
        length = len(tok.tokens) + len(isolates) + 1
        if length > self.max_length:
            raise ValueError(f"complete HIR needs {length} positions, exceeds max_length={self.max_length}; no truncation")
        roles = [int(SequenceType[token.token_type.name]) for token in tok.tokens]
        ids = [token.node_symbol if token.token_type == TokenType.NODE_REF else -1 for token in tok.tokens]
        occurrences = [list(refs) for refs in tok.symbol_occurrences]
        for u in isolates:
            occurrences[u].append(len(roles))
            roles.append(int(SequenceType.ISOLATE))
            ids.append(u)
        graph_position = len(roles)
        roles.append(int(SequenceType.GRAPH))
        ids.append(-1)
        return SequenceProgram(tuple(roles), tuple(ids), program.node_symbols,
                               tuple(tuple(rows) for rows in occurrences), tok.instruction_spans,
                               tuple(b.edge_token_position for b in tok.edge_endpoints), tok.edge_symbols,
                               tuple(b.nodes for b in tok.edge_endpoints), isolates, graph_position)

    def decode(self, sequence: SequenceProgram) -> GraphIRProgram:
        """Decode actual reference positions, retaining original node and edge rows."""
        n = len(sequence.node_symbols)
        if sequence.node_symbols != tuple(range(n)) or n > MAX_NODES:
            raise ValueError("invalid sequence node symbol table")
        if (len(sequence.node_ids) != sequence.token_length
                or sequence.graph_position != sequence.token_length-1
                or sequence.token_types[sequence.graph_position] != SequenceType.GRAPH):
            raise ValueError("invalid sequence layout")
        edges = []
        seen_nodes = set()
        if len({len(sequence.instruction_spans), len(sequence.edge_positions),
                len(sequence.edge_symbols), len(sequence.edge_endpoints)}) != 1:
            raise ValueError("edge mapping lengths do not match")
        for row, ((start, end), edge_position, symbol) in enumerate(zip(
                sequence.instruction_spans, sequence.edge_positions, sequence.edge_symbols)):
            if (start != 5*row or end != start+5 or edge_position != start or symbol != row
                    or sequence.token_types[start:end] != (0, 1, 2, 2, 3)):
                raise ValueError(f"invalid edge instruction at row {row}")
            u, v = sequence.node_ids[start+2:start+4]
            if not (0 <= u < n and 0 <= v < n) or (u, v) != sequence.edge_endpoints[row]:
                raise ValueError(f"invalid endpoint binding at edge {row}")
            edges.append(EdgeInstruction(symbol, (u, v)))
            seen_nodes.update((u, v))
        start_isolates = 5*len(edges)
        actual_isolates = sequence.node_ids[start_isolates:sequence.graph_position]
        if (actual_isolates != sequence.isolated_nodes or set(actual_isolates) != set(range(n))-seen_nodes
                or len(set(actual_isolates)) != len(actual_isolates)
                or any(role != SequenceType.ISOLATE for role in sequence.token_types[start_isolates:sequence.graph_position])):
            raise ValueError("isolated nodes are not preserved")
        occurrences = [[] for _ in range(n)]
        for position, node_id in enumerate(sequence.node_ids):
            if node_id != -1:
                if not 0 <= node_id < n:
                    raise ValueError(f"node identity outside symbol table at position {position}")
                occurrences[node_id].append(position)
        expected_positions = tuple(tuple(rows) for rows in occurrences)
        if expected_positions != sequence.node_positions:
            raise ValueError("node occurrence mapping does not match sequence identities")
        return GraphIRProgram(n, sequence.node_symbols, tuple(edges))
