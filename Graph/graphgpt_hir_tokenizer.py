"""HIR adapter for GraphGPT short/stacked tokens, without graph attributes.

Method references (independent implementation, no upstream source copied):
https://arxiv.org/abs/2401.00529v3, sections 2.2 and 2.3
https://github.com/alibaba/graph-gpt/blob/main/src/data/tokenizer/masking.py
https://github.com/alibaba/graph-gpt/blob/main/src/data/tokenizer/stacking.py

The envelope carries graph size and isolated nodes; it is not a change to IR.
Only input_ids enter the model. Symbol maps and original edge rows are metadata.
Joint identity masking in attribute-free HIR can be non-identifiable: different
hidden identities/occurrence assignments can yield exactly the same input.
First-appearance indexing also supplies ordering priors, not structural evidence.
SMTP loss/accuracy must not be interpreted as unique graph recovery quality.
"""

from dataclasses import dataclass
from enum import IntEnum
import math
import random

import torch
from torch import Tensor

from .ir import EdgeInstruction, GraphIRProgram
from .tokenizer import GraphTokenizer, TokenType

TOKENIZER_VERSION = "graphgpt_hir_stacked_v1"


class HIRType(IntEnum):
    PAD = 0
    BOS = 1
    SIZE = 2
    ISOLATE = 3
    EOS = 4
    EDGE = 5
    CONNECT = 6
    NODE_REF = 7
    END_EDGE = 8
    MASK = 9


@dataclass(frozen=True)
class HIRSequence:
    input_ids: tuple[tuple[int, int], ...]
    num_nodes: int
    # (cyclic identity, original node row), in first-appearance order.
    node_mapping: tuple[tuple[int, int], ...]
    # Original edge row for each instruction in the augmented sequence.
    edge_order: tuple[int, ...]

    @property
    def token_length(self):
        return len(self.input_ids)


@dataclass
class SMTPExample:
    input_ids: Tensor
    labels: Tensor
    masked_nodes: int
    masked_tokens: int
    identity_tokens: int
    scheduled_ratio: float


class GraphGPTHIRTokenizer:
    def __init__(self, identity_vocab_size=2048, max_length=8192,
                 tokenizer_version=TOKENIZER_VERSION):
        if tokenizer_version != TOKENIZER_VERSION:
            raise ValueError(f"Unsupported GraphGPT-HIR tokenizer: {tokenizer_version}")
        if identity_vocab_size < 1 or max_length < 3:
            raise ValueError("identity vocabulary must be positive and context length at least 3")
        self.identity_vocab_size = identity_vocab_size
        self.max_length = max_length
        self.identity_base = len(HIRType)
        self.count_base = self.identity_base + identity_vocab_size
        self.vocab_size = self.count_base + identity_vocab_size + 1
        self.config = dict(identity_vocab_size=identity_vocab_size, max_length=max_length,
                           tokenizer_version=tokenizer_version)
        self.ir_tokenizer = GraphTokenizer()

    def encode(self, program: GraphIRProgram, rng: random.Random) -> HIRSequence:
        n = program.num_nodes
        if n > self.identity_vocab_size:
            raise ValueError(f"Graph has {n} nodes, exceeds identity_vocab_size={self.identity_vocab_size}; "
                             "cyclic identities must remain injective within each graph")
        tokenized = self.ir_tokenizer(program)
        isolates = [node for node, refs in enumerate(tokenized.symbol_occurrences) if not refs]
        length = 3 + len(tokenized.tokens) + len(isolates)
        if length > self.max_length:
            raise ValueError(f"HIR requires {length} tokens (nodes={n}, edges={program.num_edges}, "
                             f"isolates={len(isolates)}), exceeds max_length={self.max_length}; "
                             "no instructions or nodes are truncated")
        # Shuffle whole instructions and flip undirected endpoints BEFORE first-appearance indexing.
        order = list(range(len(tokenized.instruction_spans)))
        rng.shuffle(order)
        offset = rng.randrange(self.identity_vocab_size)
        identities = {}
        mapping = []

        def identity(node):
            if node not in identities:
                local = (len(identities) + offset) % self.identity_vocab_size
                identities[node] = self.identity_base + local
                mapping.append((local, node))
            return identities[node]

        pairs = [(int(HIRType.BOS), 0), (int(HIRType.SIZE), self.count_base + n)]
        for row in order:
            start, end = tokenized.instruction_spans[row]
            instruction = list(tokenized.tokens[start:end])
            if rng.random() < .5:
                instruction[2], instruction[3] = instruction[3], instruction[2]
            for token in instruction:
                role = HIRType[token.token_type.name]
                value = identity(token.node_symbol) if token.token_type == TokenType.NODE_REF else 0
                pairs.append((int(role), value))
        rng.shuffle(isolates)
        for node in isolates:
            pairs.append((int(HIRType.ISOLATE), identity(node)))
        pairs.append((int(HIRType.EOS), 0))
        return HIRSequence(tuple(pairs), n, tuple(mapping), tuple(order))

    def decode(self, sequence: HIRSequence) -> GraphIRProgram:
        """Restore original node/edge rows from an unmasked sequence and its maps."""
        pairs = sequence.input_ids
        if (pairs[:2] != ((int(HIRType.BOS), 0), (int(HIRType.SIZE), self.count_base + sequence.num_nodes))
                or pairs[-1] != (int(HIRType.EOS), 0)):
            raise ValueError("invalid HIR envelope")
        mapping = dict(sequence.node_mapping)
        if (len(mapping) != sequence.num_nodes or set(mapping.values()) != set(range(sequence.num_nodes))
                or any(not 0 <= local < self.identity_vocab_size for local in mapping)):
            raise ValueError("invalid node mapping")
        count = len(sequence.edge_order)
        if sorted(sequence.edge_order) != list(range(count)):
            raise ValueError("invalid edge row mapping")
        instructions = [None] * count
        incident = set()
        for i, row in enumerate(sequence.edge_order):
            chunk = pairs[2 + 5*i:2 + 5*(i+1)]
            roles = tuple(role for role, _ in chunk)
            if roles != (HIRType.EDGE, HIRType.CONNECT, HIRType.NODE_REF, HIRType.NODE_REF, HIRType.END_EDGE):
                raise ValueError("invalid five-token HIR instruction")
            u, v = (mapping[value - self.identity_base] for _, value in chunk[2:4])
            incident.update((u, v))
            instructions[row] = EdgeInstruction(row, tuple(sorted((u, v))))
        isolated = [mapping[value - self.identity_base]
                    for role, value in pairs[2 + 5*count:-1] if role == HIRType.ISOLATE]
        if (len(pairs) != 3 + 5*count + len(isolated) or len(set(isolated)) != len(isolated)
                or set(isolated) != set(range(sequence.num_nodes)) - incident):
            raise ValueError("invalid isolated-node envelope")
        return GraphIRProgram(sequence.num_nodes, tuple(range(sequence.num_nodes)), tuple(instructions))

    def mask(self, sequence: HIRSequence, rng: random.Random) -> SMTPExample:
        # GraphGPT linear schedule: t~U[0,1), alpha(t)=1-t; sampled anew per graph,
        # NOT an epoch annealing schedule. The short-format node-level variant
        # samples ceil(alpha*N) distinct identities and masks ALL occurrences.
        ratio = 1. - rng.random()
        identities = [self.identity_base + local for local, _ in sequence.node_mapping]
        selected = set(rng.sample(identities, math.ceil(ratio * len(identities))))
        pairs, labels = [], []
        identity_tokens = masked_tokens = 0
        for role, value in sequence.input_ids:
            is_node = role in (HIRType.NODE_REF, HIRType.ISOLATE)
            masked = is_node and value in selected
            identity_tokens += int(is_node)
            masked_tokens += int(masked)
            # Keep deterministic syntax; only the identity column is a target.
            # All selected identities use MASK (no unchanged/random replacements).
            pairs.append((role, int(HIRType.MASK) if masked else value))
            labels.append(value if masked else -100)
        return SMTPExample(torch.tensor(pairs, dtype=torch.long).unsqueeze(0),
                           torch.tensor(labels, dtype=torch.long).unsqueeze(0),
                           len(selected), masked_tokens, identity_tokens, ratio)
