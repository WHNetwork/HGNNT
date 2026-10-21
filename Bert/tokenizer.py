"""Structured tokens and symbol links derived directly from DIRECT-only H-IR."""

from dataclasses import dataclass
from enum import IntEnum

from .hir import HIRProgram


class TokenType(IntEnum):
    EDGE = 0
    ORDER = 1
    DIRECT = 2
    NODE_REF = 3
    END_EDGE = 4
    PAD = 5
    MASK_REF = 6


@dataclass(frozen=True)
class HIRToken:
    token_type: TokenType
    instruction_index: int
    edge_symbol: int
    order: int
    node_symbol: int | None = None


@dataclass(frozen=True)
class TokenizedProgram:
    """Spans are [start, end); occurrences index the flat token tuple.

    symbol_occurrences aligns with node_symbols, including isolated symbols.
    order_groups stores (order, instruction indices), retaining program order.
    """

    tokens: tuple[HIRToken, ...]
    instruction_spans: tuple[tuple[int, int], ...]
    node_symbols: tuple[int, ...]
    symbol_occurrences: tuple[tuple[int, ...], ...]
    order_groups: tuple[tuple[int, tuple[int, ...]], ...]
    num_hyperedges: int


class HIRTokenizer:
    """Serialize each instruction as EDGE ORDER DIRECT NODE_REF* END_EDGE.

    Symbol IDs are metadata for alignment, never vocabulary entries. No original
    hypergraph, node features, trained vocabulary or text parser is needed.
    """

    def __call__(self, program: HIRProgram) -> TokenizedProgram:
        tokens, spans = [], []
        occurrences = {symbol: [] for symbol in program.node_symbols}
        order_groups = {}
        for index, instruction in enumerate(program.instructions):
            start = len(tokens)
            edge, order = instruction.edge_symbol, instruction.order
            for token_type in (TokenType.EDGE, TokenType.ORDER, TokenType.DIRECT):
                tokens.append(HIRToken(token_type, index, edge, order))
            for symbol in instruction.members:
                occurrences[symbol].append(len(tokens))
                tokens.append(HIRToken(TokenType.NODE_REF, index, edge, order, symbol))
            tokens.append(HIRToken(TokenType.END_EDGE, index, edge, order))
            spans.append((start, len(tokens)))
            order_groups.setdefault(order, []).append(index)
        return TokenizedProgram(
            tokens=tuple(tokens),
            instruction_spans=tuple(spans),
            node_symbols=program.node_symbols,
            symbol_occurrences=tuple(tuple(occurrences[s]) for s in program.node_symbols),
            order_groups=tuple((order, tuple(indices))
                               for order, indices in order_groups.items()),
            num_hyperedges=program.num_hyperedges,
        )
