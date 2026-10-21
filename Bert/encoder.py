"""HIR-BERT v2: local instructions, aligned symbol context, local refinement."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .embedding import HIREmbedding
from .hir import HIRProgram
from .tokenizer import HIRTokenizer, TokenizedProgram, TokenType


@dataclass
class EncoderOutput:
    node_embeddings: Tensor
    hyperedge_embeddings: Tensor
    graph_embedding: Tensor


class HIRBertEncoder(nn.Module):
    """Optional node features; output rows follow source node/edge coordinates.

    Both local stages batch complete instructions of equal order. Symbol context
    spans every occurrence regardless of instruction batches. No absolute
    positions or numeric identity embeddings are used; member order has no
    semantic role. num_layers controls each local Transformer's depth.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1,
                 instruction_batch_size: int = 32, feature_dim: int | None = None):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_layers < 1 or instruction_batch_size < 1:
            raise ValueError("num_layers and instruction_batch_size must be positive")
        self.config = dict(hidden_dim=hidden_dim, num_layers=num_layers,
                           num_heads=num_heads, dropout=dropout,
                           instruction_batch_size=instruction_batch_size,
                           feature_dim=feature_dim)
        self.hidden_dim = hidden_dim
        self.instruction_batch_size = instruction_batch_size
        self.tokenizer = HIRTokenizer()
        self.embedding = HIREmbedding(hidden_dim, feature_dim=feature_dim)
        self.local_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, 4 * hidden_dim,
                                       dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.refinement_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden_dim, num_heads, 4 * hidden_dim,
                                       dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.symbol_projection = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.symbol_norm = nn.LayerNorm(hidden_dim)
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.feedback_norm = nn.LayerNorm(hidden_dim)

    def _encode_instructions(self, states: Tensor, program: TokenizedProgram,
                             layers: nn.ModuleList) -> Tensor:
        """Apply shared layers to whole instructions, without padding or chunks."""
        encoded = [None] * len(program.instruction_spans)
        for _, indices in program.order_groups:
            for start in range(0, len(indices), self.instruction_batch_size):
                batch_indices = indices[start:start + self.instruction_batch_size]
                spans = [program.instruction_spans[index] for index in batch_indices]
                batch = torch.stack([states[s:e] for s, e in spans])
                for layer in layers:
                    batch = layer(batch)
                for row, index in enumerate(batch_indices):
                    encoded[index] = batch[row]
        return torch.cat(encoded, dim=0) if encoded else states

    def _symbol_context(self, states: Tensor, occurrences: tuple[Tensor, ...]) -> Tensor:
        """Sum all occurrences, then apply a biased projection and LayerNorm."""
        contexts = []
        for positions in occurrences:
            if positions.numel():
                references = states.index_select(0, positions)
                context = references.sum(dim=0)
            else:
                # Isolated symbols have no instruction context. Use the shared
                # NODE_REF embedding without inventing an incidence.
                context = self.embedding.syntax.weight[TokenType.NODE_REF]
            contexts.append(context)
        pooled = torch.stack(contexts) if contexts else states.new_empty((0, self.hidden_dim))
        # The bias lets multiplicity affect the projected direction before norm.
        return self.symbol_norm(self.symbol_projection(pooled))

    def forward(self, program: HIRProgram, x: Tensor | None = None) -> EncoderOutput:
        tokenized = self.tokenizer(program)
        states = self.embedding(tokenized, x=x)
        device = states.device
        occurrences = tuple(torch.tensor(positions, dtype=torch.long, device=device)
                            for positions in tokenized.symbol_occurrences)

        # Stage 1: preserve a contextual state for every token occurrence.
        local = self._encode_instructions(states, tokenized, self.local_layers)

        # Stage 2: align by symbol, sum across instructions, then feed the
        # shared context back through a residual to preserve each local state.
        symbols = self._symbol_context(local, occurrences)
        positions = torch.cat(occurrences) if occurrences else torch.empty(
            0, dtype=torch.long, device=device)
        symbol_rows = torch.tensor(
            [row for row, refs in enumerate(tokenized.symbol_occurrences) for _ in refs],
            dtype=torch.long, device=device)
        references = local.index_select(0, positions)
        updated = self.feedback_norm(
            references + self.context_projection(symbols.index_select(0, symbol_rows)))
        feedback = local.index_copy(0, positions, updated)

        # Stage 3: other instruction tokens now attend to updated NODE_REFs.
        refined = self._encode_instructions(feedback, tokenized, self.refinement_layers)
        symbol_context = self._symbol_context(refined, occurrences)
        node_rows = torch.tensor(tokenized.node_symbols, dtype=torch.long, device=device)
        nodes = refined.new_zeros((len(node_rows), self.hidden_dim)).index_copy(
            0, node_rows, symbol_context)
        edge_positions = torch.tensor([start for start, _ in tokenized.instruction_spans],
                                      dtype=torch.long, device=device)
        edge_rows = torch.tensor([tokenized.tokens[start].edge_symbol
                                  for start, _ in tokenized.instruction_spans],
                                 dtype=torch.long, device=device)
        edges = refined.new_zeros((tokenized.num_hyperedges, self.hidden_dim)).index_copy(
            0, edge_rows, refined.index_select(0, edge_positions))
        graph = (nodes.sum(dim=0) + edges.sum(dim=0)) / max(len(nodes) + len(edges), 1)
        return EncoderOutput(nodes, edges, graph)


class HIRGuidedBlock(nn.Module):
    """Instruction-local attention followed by symbol-bound cross-attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.instruction_attention = nn.TransformerEncoderLayer(
            hidden_dim, num_heads, 4 * hidden_dim, dropout, batch_first=True)
        self.symbol_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.symbol_norm = nn.LayerNorm(hidden_dim)
        self.symbol_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim))
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedback_projection = nn.Linear(hidden_dim, hidden_dim)
        self.feedback_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: Tensor, symbols: Tensor, program: TokenizedProgram,
                symbol_groups: tuple, reference_positions: Tensor,
                reference_rows: Tensor, batch_size: int) -> tuple[Tensor, Tensor]:
        # A: no instruction attends to another instruction's tokens.
        instructions = [None] * len(program.instruction_spans)
        for _, indices in program.order_groups:
            for start in range(0, len(indices), batch_size):
                rows = indices[start:start + batch_size]
                spans = [program.instruction_spans[row] for row in rows]
                batch = self.instruction_attention(torch.stack([tokens[s:e] for s, e in spans]))
                for row, index in enumerate(rows):
                    instructions[index] = batch[row]
        tokens = torch.cat(instructions, dim=0) if instructions else tokens

        # B: each query sees only its own NODE_REFs. Equal-length groups batch
        # independent queries without a global mask or padded attention matrix.
        context = torch.zeros_like(symbols)
        for symbol_rows, positions in symbol_groups:
            for start in range(0, len(symbol_rows), batch_size):
                rows = symbol_rows[start:start + batch_size]
                refs = positions[start:start + batch_size]
                query = symbols.index_select(0, rows).unsqueeze(1)
                references = tokens[refs]
                attended, _ = self.symbol_attention(
                    query, references, references, need_weights=False)
                context = context.index_copy(0, rows, attended.squeeze(1))
        # Isolated symbols keep their own residual and FFN, with no fake refs.
        symbols = self.symbol_norm(symbols + self.dropout(context))
        symbols = self.ffn_norm(symbols + self.dropout(self.symbol_ffn(symbols)))

        # C: retain every reference's local context while sharing symbol state.
        references = tokens.index_select(0, reference_positions)
        feedback = self.feedback_projection(symbols.index_select(0, reference_rows))
        references = self.feedback_norm(references + self.dropout(feedback))
        tokens = tokens.index_copy(0, reference_positions, references)
        return tokens, symbols


class HIRGuidedEncoder(nn.Module):
    """Persistent symbol states guide every block through grouped attention.

    No global token attention, node-ID embeddings or positional encodings are
    used. instruction_batch_size batches whole instructions and symbol groups.
    Feature-aware symbols share the NODE_REF feature projection; structure-only
    symbols start from the shared NODE_REF syntax embedding.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.1,
                 instruction_batch_size: int = 32, feature_dim: int | None = None):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if num_layers < 1 or instruction_batch_size < 1:
            raise ValueError("num_layers and instruction_batch_size must be positive")
        self.config = dict(hidden_dim=hidden_dim, num_layers=num_layers,
                           num_heads=num_heads, dropout=dropout,
                           instruction_batch_size=instruction_batch_size,
                           feature_dim=feature_dim)
        self.hidden_dim = hidden_dim
        self.instruction_batch_size = instruction_batch_size
        self.tokenizer = HIRTokenizer()
        self.embedding = HIREmbedding(hidden_dim, feature_dim=feature_dim)
        self.blocks = nn.ModuleList([
            HIRGuidedBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, program: HIRProgram, x: Tensor | None = None) -> EncoderOutput:
        tokenized = self.tokenizer(program)
        tokens = self.embedding(tokenized, x=x)
        device = tokens.device
        node_rows = torch.tensor(tokenized.node_symbols, dtype=torch.long, device=device)
        if x is not None:
            features = x.to(device=device, dtype=tokens.dtype)
            symbols = self.embedding.feature_projection(features.index_select(0, node_rows))
        else:
            symbols = self.embedding.syntax.weight[TokenType.NODE_REF].expand(len(node_rows), -1)

        # These links come exclusively from the tokenizer, not hyperedge_index.
        groups = {}
        for row, positions in enumerate(tokenized.symbol_occurrences):
            if positions:
                groups.setdefault(len(positions), []).append((row, positions))
        symbol_groups = tuple(
            (torch.tensor([row for row, _ in group], dtype=torch.long, device=device),
             torch.tensor([positions for _, positions in group], dtype=torch.long, device=device))
            for group in groups.values())
        reference_positions = torch.tensor(
            [pos for positions in tokenized.symbol_occurrences for pos in positions],
            dtype=torch.long, device=device)
        reference_rows = torch.tensor(
            [row for row, positions in enumerate(tokenized.symbol_occurrences) for _ in positions],
            dtype=torch.long, device=device)
        for block in self.blocks:
            tokens, symbols = block(tokens, symbols, tokenized, symbol_groups,
                                    reference_positions, reference_rows, self.instruction_batch_size)

        nodes = tokens.new_zeros((len(node_rows), self.hidden_dim)).index_copy(0, node_rows, symbols)
        edge_positions = torch.tensor([start for start, _ in tokenized.instruction_spans],
                                      dtype=torch.long, device=device)
        edge_rows = torch.tensor([tokenized.tokens[start].edge_symbol
                                  for start, _ in tokenized.instruction_spans],
                                 dtype=torch.long, device=device)
        edges = tokens.new_zeros((tokenized.num_hyperedges, self.hidden_dim)).index_copy(
            0, edge_rows, tokens.index_select(0, edge_positions))
        graph = (nodes.sum(dim=0) + edges.sum(dim=0)) / max(len(nodes) + len(edges), 1)
        return EncoderOutput(nodes, edges, graph)
