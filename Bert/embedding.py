"""Learnable syntax and hyperedge-order embeddings for structured H-IR tokens."""

import torch
from torch import Tensor, nn

from .tokenizer import TokenizedProgram, TokenType


class HIREmbedding(nn.Module):
    """No symbol vocabulary, position encoding or handcrafted node features."""

    def __init__(self, hidden_dim: int = 128, feature_dim: int | None = None):
        super().__init__()
        self.syntax = nn.Embedding(len(TokenType), hidden_dim)
        self.order_projection = nn.Linear(1, hidden_dim)
        self.feature_projection = (nn.Linear(feature_dim, hidden_dim)
                                   if feature_dim is not None else None)

    def forward(self, program: TokenizedProgram, x: Tensor | None = None) -> Tensor:
        device = self.syntax.weight.device
        types = torch.tensor([token.token_type for token in program.tokens],
                             dtype=torch.long, device=device)
        orders = self.syntax.weight.new_tensor([token.order for token in program.tokens])
        embeddings = self.syntax(types) + self.order_projection(orders.log1p().unsqueeze(-1))
        if x is not None:
            if self.feature_projection is None:
                raise ValueError("feature_dim must be configured when supplying node features")
            positions = torch.tensor(
                [i for i, token in enumerate(program.tokens)
                 if token.token_type == TokenType.NODE_REF], dtype=torch.long, device=device)
            symbols = torch.tensor(
                [token.node_symbol for token in program.tokens
                 if token.token_type == TokenType.NODE_REF], dtype=torch.long, device=device)
            features = self.feature_projection(x.to(device=device, dtype=self.syntax.weight.dtype))
            embeddings = embeddings.index_add(0, positions, features.index_select(0, symbols))
        return embeddings
