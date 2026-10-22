"""Native ordinary-graph intermediate representation learning."""

from .ir import EdgeInstruction, GraphIRProgram, compile_graph, to_edge_index
from .tokenizer import GraphTokenizer

__all__ = ["EdgeInstruction", "GraphIRProgram", "compile_graph", "to_edge_index", "GraphTokenizer"]
