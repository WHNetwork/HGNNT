"""H-IR-BERT V0.1: structural BERT encoder and supervised node readout."""

import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from hir import (EDGE_ID, NODE, NUM_TYPES, ORDER, PAD,
                 chunk_hir, compile_hypergraph)


HIRBERT_DEFAULTS = {
    "hidden_size": 256, "num_hidden_layers": 4, "num_attention_heads": 4,
    "intermediate_size": 1024, "dropout": 0.2, "max_seq_length": 256,
    "chunk_batch_size": 8, "gradient_checkpointing": True,
}


class HIREncoder(nn.Module):
    """Feature-aware structural encoder, independent of classification loss."""

    def __init__(self, num_features, config):
        super().__init__()
        # Import only when HIRBERT is constructed; other methods need no HF dependency.
        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:
            raise ImportError(
                'HIRBERT requires transformers. Install transformers==4.44.2 '
                '(with tokenizers==0.19.1); keep the existing PyTorch/CUDA build.'
            ) from exc
        hidden = config["hidden_size"]
        self.chunk_batch_size = config["chunk_batch_size"]
        self.use_checkpointing = config["gradient_checkpointing"]
        if self.chunk_batch_size < 1 or config["max_seq_length"] < 6:
            raise ValueError("chunk_batch_size >= 1 and max_seq_length >= 6 are required")
        bert_config = BertConfig(
            vocab_size=NUM_TYPES, hidden_size=hidden,
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            hidden_dropout_prob=config["dropout"],
            attention_probs_dropout_prob=config["dropout"],
            max_position_embeddings=config["max_seq_length"],
            type_vocab_size=1, pad_token_id=PAD, attn_implementation="eager",
        )
        # Random initialization; no tokenizer, pretrained weights or network access.
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.feature_projection = nn.Linear(num_features, hidden)
        self.order_projection = nn.Linear(1, hidden, bias=False)
        self.register_buffer(
            "frequencies", torch.exp(-math.log(10000.0) * torch.arange(0, hidden, 2) / hidden),
            persistent=False,
        )

    def reset_parameters(self):
        self.bert.apply(self.bert._init_weights)
        self.feature_projection.reset_parameters()
        self.order_projection.reset_parameters()

    def _encode_readouts(self, inputs, mask, rows, positions):
        hidden = self.bert(inputs_embeds=inputs, attention_mask=mask,
                           return_dict=True).last_hidden_state
        return hidden[rows, positions]

    def encode_node_queries(self, projected_features, node_ids, batch_size=256):
        """Encode candidate NODE queries without supplying any membership.

        Pretraining uses these for candidates with no remaining visible context.
        The BERT, NODE embedding, sinusoidal IDs and feature projection are the
        same ones used by forward(). No new encoder parameters are introduced.
        """
        states = []
        hidden = projected_features.shape[-1]
        for ids in node_ids.split(batch_size):
            phase = ids.to(projected_features.dtype).unsqueeze(-1) * self.frequencies
            identity = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :hidden]
            types = torch.full_like(ids, NODE)
            inputs = (self.bert.embeddings.word_embeddings(types) + identity
                      + projected_features[ids]).unsqueeze(1)
            mask = torch.ones((len(ids), 1), dtype=torch.long, device=ids.device)
            args = (inputs, mask, torch.arange(len(ids), device=ids.device), torch.zeros_like(ids))
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                encoded = checkpoint(self._encode_readouts, *args, use_reentrant=False)
            else:
                encoded = self._encode_readouts(*args)
            states.append(encoded)
        return torch.cat(states) if states else projected_features.new_empty((0, hidden))

    def forward(self, projected_features, program, chunks):
        hidden = projected_features.shape[1]
        num_edges = len(program.hyperedges)
        if not chunks:
            return projected_features.new_zeros((num_edges, hidden))
        device, space = projected_features.device, program.space
        # A zero row supplies feature-free syntax positions, including empty edges.
        features = torch.cat((projected_features, projected_features.new_zeros((1, hidden))))
        states, edge_ids, weights = [], [], []
        for start in range(0, len(chunks), self.chunk_batch_size):
            batch = chunks[start:start + self.chunk_batch_size]
            length = max(len(c.tokens) for c in batch)
            raw = torch.full((len(batch), length), PAD, dtype=torch.long, device=device)
            orders = torch.zeros_like(raw)
            mask = torch.zeros_like(raw)
            locations, targets, contributions = [], [], []
            for row, chunk in enumerate(batch):
                size = len(chunk.tokens)
                raw[row, :size] = torch.tensor(chunk.tokens, device=device)
                orders[row, :size] = torch.tensor(chunk.orders, device=device)
                mask[row, :size] = 1
                for position, target, weight in chunk.readouts:
                    locations.append((row, position))
                    targets.append(target)
                    contributions.append(weight)
            is_node = (raw >= space.node(0)) & (raw < space.edge(0))
            is_edge = (raw >= space.edge(0)) & (raw < space.order(0))
            is_order = raw >= space.order(0)
            types = torch.where(is_node, NODE, raw)
            types = torch.where(is_edge, EDGE_ID, types)
            types = torch.where(is_order, ORDER, types)
            payload = torch.where(is_node, raw - space.node(0), torch.zeros_like(raw))
            payload = torch.where(is_edge, raw - space.edge(0), payload)
            payload = torch.where(is_order, raw - space.order(0), payload)
            phase = payload.unsqueeze(-1).to(projected_features.dtype) * self.frequencies
            identity = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)[..., :hidden]
            identity = identity * (is_node | is_edge | is_order).unsqueeze(-1)
            node_ids = torch.where(is_node, raw - space.node(0), program.num_nodes)
            inputs = (self.bert.embeddings.word_embeddings(types) + identity + features[node_ids]
                      + self.order_projection(orders.to(projected_features.dtype).log1p().unsqueeze(-1)))
            locations = torch.tensor(locations, dtype=torch.long, device=device)
            args = (inputs, mask, locations[:, 0], locations[:, 1])
            if self.use_checkpointing and self.training and torch.is_grad_enabled():
                # Return only readouts, and recompute each bounded BERT batch in backward.
                encoded = checkpoint(self._encode_readouts, *args, use_reentrant=False)
            else:
                encoded = self._encode_readouts(*args)
            states.append(encoded)
            edge_ids.append(torch.tensor(targets, dtype=torch.long, device=device))
            weights.append(torch.tensor(contributions, dtype=encoded.dtype, device=device))
        encoded_states = torch.cat(states)
        return encoded_states.new_zeros((num_edges, hidden)).index_add(
            0, torch.cat(edge_ids), encoded_states * torch.cat(weights).unsqueeze(-1))


class HIRBERT(nn.Module):
    def __init__(self, num_features, num_classes, args):
        super().__init__()
        self.model_config = {}
        for key, default in HIRBERT_DEFAULTS.items():
            value = getattr(args, "hir_" + key, None)
            self.model_config[key] = default if value is None else value
        self.num_features, self.num_classes = num_features, num_classes
        self.encoder = HIREncoder(num_features, self.model_config)
        hidden = self.model_config["hidden_size"]
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Dropout(self.model_config["dropout"]), nn.Linear(hidden, num_classes),
        )
        self._source_index = None
        self.program = self.chunks = None
        self.compilation_stats = {}

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()

    def prepare(self, data):
        """Cache only deterministic structure, never BERT outputs or features."""
        index = data.hyperedge_index
        declared_edges = getattr(data, 'num_hyperedges', None)
        declared_edges = None if declared_edges is None else int(declared_edges)
        if (self._source_index is index and self._source_version == index._version
                and self.program.num_nodes == data.x.shape[0]
                and self._declared_edges == declared_edges):
            return
        cpu_index = index.detach().cpu()
        explicit_count = declared_edges
        if declared_edges is not None and cpu_index.numel() and int(cpu_index[1].max()) >= declared_edges:
            # DHG-Bench also accepts noncontiguous source IDs with a count of
            # represented edges. Explicit empty columns need dense source IDs.
            if len(cpu_index[1].unique()) != declared_edges:
                raise ValueError('Explicit empty edges require source IDs in [0, num_hyperedges)')
            explicit_count = None
        self.program = compile_hypergraph(data.x.shape[0], cpu_index, explicit_count)
        self.chunks, chunk_stats = chunk_hir(self.program, self.model_config["max_seq_length"])
        self.compilation_stats = {**self.program.stats, **chunk_stats}
        self._incidence = self.program.to_hyperedge_index()
        self._source_index, self._source_version = index, index._version
        self._declared_edges = declared_edges
        print("[HIRBERT compilation]", self.compilation_stats)

    def forward(self, data):
        self.prepare(data)
        features = self.encoder.feature_projection(data.x)
        edges = self.encoder(features, self.program, self.chunks)
        node_ids, edge_ids = self._incidence.to(features.device)
        messages = torch.zeros_like(features).index_add(0, node_ids, edges[edge_ids])
        degrees = torch.bincount(node_ids, minlength=features.shape[0]).clamp_min(1)
        messages = messages / degrees.unsqueeze(-1)
        logits = self.classifier(torch.cat((features, messages), dim=-1))
        return logits, None

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        """Load a final checkpoint and its graph for standalone inference."""
        saved = torch.load(path, map_location=device, weights_only=True)
        args = SimpleNamespace(**{"hir_" + k: v for k, v in saved["model_config"].items()})
        model = cls(saved["num_features"], saved["num_classes"], args).to(device)
        model.load_state_dict(saved["model_state_dict"])
        model.eval()
        data = SimpleNamespace(**saved["graph"])
        return model, data
