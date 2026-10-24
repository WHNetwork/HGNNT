"""Bidirectional HIR representation encoder, without a token-prediction head.

Reuses Hugging Face Llama (Apache-2.0, EleutherAI/Hugging Face contributors),
following the explicit noncausal 4D-mask API already used by GraphGPT-HIR.
Dependency: transformers==4.53.3. No GraphGPT/Sparse weights are converted.
"""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
from transformers import LlamaConfig, LlamaModel
from transformers.utils.versions import require_version

from .encoder import EncoderOutput
from .sequence_tokenizer import HIRSequenceTokenizer, MAX_NODES, SequenceProgram, SequenceType

ARCHITECTURE_VERSION = "hir_bidirectional_sequence_v2"
INITIALIZATION_VERSION = "llama_scaled_id_v1"
MODEL_SIZES = dict(small=dict(hidden_dim=256, num_layers=4, num_heads=4, intermediate_size=1024),
                   base=dict(hidden_dim=768, num_layers=12, num_heads=12, intermediate_size=3072))


class HexIDEmbedding(nn.Module):
    """Position-specific digits; shared prefixes still correlate different IDs.

    In particular, 0000--000F share three projected digit components. Matching
    the RMS scale does not remove that correlation or change Hex semantics.
    """
    def __init__(self, hidden_dim, digit_dim=32, initializer_range=0.02):
        super().__init__()
        self.digits = nn.ModuleList([nn.Embedding(16, digit_dim) for _ in range(4)])
        self.projection = nn.Linear(4*digit_dim, hidden_dim, bias=False)
        for table in self.digits:
            nn.init.normal_(table.weight, mean=0., std=1.)
        nn.init.normal_(self.projection.weight, mean=0.,
                        std=initializer_range / math.sqrt(4*digit_dim))

    def forward(self, ids):
        if ids.dtype != torch.long or ((ids < 0) | (ids >= MAX_NODES)).any():
            raise ValueError("hex IDs must be LongTensor values in [0, 65535]")
        shifts = ids.new_tensor([12, 8, 4, 0])
        values = (ids.unsqueeze(-1) >> shifts) & 15
        return self.projection(torch.cat([table(values[..., i]) for i, table in enumerate(self.digits)], dim=-1))


@dataclass
class SequenceEncoding:
    sequence: SequenceProgram
    initial_embeddings: Tensor
    last_hidden_state: Tensor
    # HF includes the input and each layer; only the final layer has final RMSNorm.
    hidden_states: tuple[Tensor, ...] | None


class HIRSequenceEncoder(nn.Module):
    def __init__(self, size="base", id_encoding="hex", digit_dim=32, feature_dim=None,
                 max_length=8192, attention_dropout=0., architecture_version=ARCHITECTURE_VERSION,
                 initialization_version=INITIALIZATION_VERSION):
        super().__init__()
        require_version("transformers==4.53.3", "HIRSequenceEncoder uses its explicit 4D noncausal mask API.")
        if architecture_version != ARCHITECTURE_VERSION:
            raise ValueError(f"incompatible sequence architecture {architecture_version!r}; expected {ARCHITECTURE_VERSION!r}")
        if initialization_version != INITIALIZATION_VERSION:
            raise ValueError(f"incompatible ID initialization {initialization_version!r}; expected {INITIALIZATION_VERSION!r}")
        if size not in MODEL_SIZES or id_encoding not in ("hex", "lookup"):
            raise ValueError("size must be small/base and id_encoding must be hex/lookup")
        if digit_dim < 1 or (feature_dim is not None and feature_dim < 1) or not 0 <= attention_dropout < 1:
            raise ValueError("invalid digit/feature dimension or attention dropout")
        self.config = dict(size=size, id_encoding=id_encoding, digit_dim=digit_dim, feature_dim=feature_dim,
                           max_length=max_length, attention_dropout=attention_dropout,
                           architecture_version=architecture_version, initialization_version=initialization_version)
        dimensions = MODEL_SIZES[size]
        self.hidden_dim = dimensions["hidden_dim"]
        self.tokenizer = HIRSequenceTokenizer(max_length)
        config = LlamaConfig(vocab_size=len(SequenceType), hidden_size=self.hidden_dim,
                             num_hidden_layers=dimensions["num_layers"],
                             num_attention_heads=dimensions["num_heads"], num_key_value_heads=dimensions["num_heads"],
                             intermediate_size=dimensions["intermediate_size"], max_position_embeddings=max_length,
                             pad_token_id=int(SequenceType.PAD), bos_token_id=None, eos_token_id=None,
                             attention_dropout=attention_dropout, use_cache=False, tie_word_embeddings=False)
        config._attn_implementation = "sdpa"
        self.backbone = LlamaModel(config)
        for layer in self.backbone.layers:
            layer.self_attn.is_causal = False
        if id_encoding == "hex":
            self.id_embedding = HexIDEmbedding(self.hidden_dim, digit_dim,
                                               initializer_range=config.initializer_range)
        else:
            self.id_embedding = nn.Embedding(MAX_NODES, self.hidden_dim)
            nn.init.normal_(self.id_embedding.weight, mean=0., std=config.initializer_range)
        self.feature_projection = nn.Linear(feature_dim, self.hidden_dim, bias=False) if feature_dim is not None else None
        self.node_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

    def embed(self, sequence, x=None):
        """One type embedding plus the same ID/optional attribute vector per reference."""
        weight = self.backbone.embed_tokens.weight
        device = weight.device
        types = torch.tensor(sequence.token_types, dtype=torch.long, device=device)
        states = self.backbone.embed_tokens(types)
        n = len(sequence.node_symbols)
        ids = torch.tensor(sequence.node_symbols, dtype=torch.long, device=device)
        node_inputs = self.id_embedding(ids)
        if x is not None:
            if self.feature_projection is None:
                raise ValueError("x was supplied but feature_dim is not configured")
            if x.ndim != 2 or x.shape != (n, self.config["feature_dim"]):
                raise ValueError("x must have shape [num_nodes, feature_dim]")
            node_inputs = node_inputs + self.feature_projection(x.to(device=device, dtype=weight.dtype))
        positions, owners = [], []
        for u, rows in enumerate(sequence.node_positions):
            positions.extend(rows)
            owners.extend([u]*len(rows))
        positions = torch.tensor(positions, dtype=torch.long, device=device)
        owners = torch.tensor(owners, dtype=torch.long, device=device)
        # Autocast can make the Hex projection BF16 while type lookup stays
        # FP32. index_add requires identical source/destination dtypes.
        return states.index_add(0, positions, node_inputs[owners].to(dtype=states.dtype))

    def encode_tokens(self, program, x=None, output_hidden_states=False):
        sequence = self.tokenizer(program)
        initial = self.embed(sequence, x)
        length = sequence.token_length
        # A prepared 4D mask bypasses create_causal_mask in pinned Transformers.
        # Every entry is zero, including above the diagonal; there is no padding
        # in this single-graph interface and no Reference Graph restriction.
        mask = initial.new_zeros((1, 1, 1, length)).expand(1, 1, length, length)
        positions = torch.arange(length, device=initial.device).unsqueeze(0)
        output = self.backbone(inputs_embeds=initial.unsqueeze(0), attention_mask=mask,
                               position_ids=positions, use_cache=False, return_dict=True,
                               output_hidden_states=output_hidden_states)
        hidden = tuple(state.squeeze(0) for state in output.hidden_states) if output_hidden_states else None
        return SequenceEncoding(sequence, initial, output.last_hidden_state.squeeze(0), hidden)

    def readout(self, sequence, states):
        """Mean of contextual references (or ISOLATE), then one shared projection."""
        n, d = len(sequence.node_symbols), self.hidden_dim
        positions, owners = [], []
        for u, rows in enumerate(sequence.node_positions):
            positions.extend(rows)
            owners.extend([u]*len(rows))
        positions = torch.tensor(positions, dtype=torch.long, device=states.device)
        owners = torch.tensor(owners, dtype=torch.long, device=states.device)
        nodes = states.new_zeros((n, d)).index_add(0, owners, states[positions])
        counts = states.new_tensor([len(rows) for rows in sequence.node_positions]).unsqueeze(-1)
        nodes = self.node_projection(nodes/counts.clamp_min(1))
        edges = torch.tensor(sequence.edge_positions, dtype=torch.long, device=states.device)
        return EncoderOutput(nodes, states[edges], states[sequence.graph_position])

    def forward(self, program, x=None):
        encoded = self.encode_tokens(program, x)
        return self.readout(encoded.sequence, encoded.last_hidden_state)

    def checkpoint_dict(self):
        return dict(architecture_version=ARCHITECTURE_VERSION, initialization_version=INITIALIZATION_VERSION,
                    encoder_config=dict(self.config),
                    tokenizer_config=dict(self.tokenizer.config), encoder_state_dict=self.state_dict())

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        config = checkpoint.get("encoder_config", {})
        if (checkpoint.get("architecture_version") != ARCHITECTURE_VERSION
                or config.get("architecture_version") != ARCHITECTURE_VERSION):
            raise ValueError("incompatible checkpoint: expected HIRSequenceEncoder v2 with scaled ID initialization; "
                             "old sequence v1, Sparse and GraphGPT-HIR SMTP weights are unsupported")
        if (checkpoint.get("initialization_version") != INITIALIZATION_VERSION
                or config.get("initialization_version") != INITIALIZATION_VERSION):
            raise ValueError(f"checkpoint must explicitly declare initialization_version={INITIALIZATION_VERSION!r}")
        model = cls(**config)
        if checkpoint.get("tokenizer_config") != model.tokenizer.config:
            raise ValueError("checkpoint tokenizer configuration does not match the sequence encoder")
        model.load_state_dict(checkpoint["encoder_state_dict"], strict=True)
        return model.to(device)
