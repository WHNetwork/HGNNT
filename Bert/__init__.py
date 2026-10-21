"""HIR-BERT v2: structured H-IR tokens and symbol-aligned contextual encoding."""

from .hir import EdgeInstruction, HIRProgram, compile_hypergraph
from .tokenizer import HIRToken, HIRTokenizer, TokenizedProgram, TokenType
from .embedding import HIREmbedding
from .encoder import EncoderOutput, HIRBertEncoder, HIRGuidedEncoder

__all__ = [
    "EdgeInstruction", "HIRProgram", "compile_hypergraph",
    "TokenType", "HIRToken", "TokenizedProgram", "HIRTokenizer", "HIREmbedding",
    "EncoderOutput", "HIRBertEncoder", "HIRGuidedEncoder",
]
