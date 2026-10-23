"""GraphGPT-HIR: stacked token lookup + bidirectional Llama + SMTP CE head.

Dependency: transformers==4.53.3 (the version used by official GraphGPT).
No upstream implementation is vendored. Llama components are imported from
Hugging Face Transformers (Apache-2.0, EleutherAI/Hugging Face contributors).
Method attribution: Alibaba GraphGPT, https://github.com/alibaba/graph-gpt,
src/models/graphgpt/{modeling_common,modeling_helpers,utils_graphgpt}.py.
The official sum aggregation and unshifted masked-token CE are retained; HIR
has two stacked columns and supervises only node identities, not attributes.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel
from transformers.utils.versions import require_version

MODEL_VERSION = "graphgpt_hir_smtp_v1"


@dataclass
class GraphGPTHIROutput:
    hidden_states: Tensor
    logits: Tensor
    loss: Tensor | None


class GraphGPTHIR(nn.Module):
    def __init__(self, vocab_size, max_length=8192, hidden_dim=256, num_layers=4,
                 num_heads=4, intermediate_size=1024, attention_dropout=0.,
                 model_version=MODEL_VERSION):
        super().__init__()
        require_version("transformers==4.53.3", "GraphGPT-HIR uses the reviewed Llama 4.53.3 mask API.")
        if model_version != MODEL_VERSION:
            raise ValueError(f"Unsupported GraphGPT-HIR model version: {model_version}")
        if (hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads
                or (hidden_dim // num_heads) % 2 or num_layers < 1 or intermediate_size < 1
                or vocab_size < 1 or max_length < 3 or not 0 <= attention_dropout < 1):
            raise ValueError("invalid Llama dimensions, RoPE head width, context length, or dropout")
        self.config = dict(vocab_size=vocab_size, max_length=max_length, hidden_dim=hidden_dim,
                           num_layers=num_layers, num_heads=num_heads, intermediate_size=intermediate_size,
                           attention_dropout=attention_dropout, model_version=model_version)
        config = LlamaConfig(vocab_size=vocab_size, hidden_size=hidden_dim,
                             intermediate_size=intermediate_size, num_hidden_layers=num_layers,
                             num_attention_heads=num_heads, num_key_value_heads=num_heads,
                             max_position_embeddings=max_length, attention_dropout=attention_dropout,
                             pad_token_id=0, bos_token_id=1, eos_token_id=4,
                             use_cache=False, tie_word_embeddings=False, hidden_act="silu")
        config._attn_implementation = "sdpa"
        self.backbone = LlamaModel(config)
        for layer in self.backbone.layers:
            layer.self_attn.is_causal = False
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        nn.init.normal_(self.lm_head.weight, std=config.initializer_range)

    def encode(self, input_ids, attention_mask=None):
        """Return token states for pretraining or a future downstream task head."""
        if input_ids.ndim != 3 or input_ids.shape[-1] != 2:
            raise ValueError("input_ids must have shape [batch, length, 2]")
        batch, length, _ = input_ids.shape
        if length > self.config["max_length"]:
            raise ValueError(f"HIR length {length} exceeds max_length={self.config['max_length']}; no truncation")
        if attention_mask is None:
            attention_mask = input_ids[:, :, 0] != 0
        if attention_mask.shape != (batch, length):
            raise ValueError("attention_mask must have shape [batch, length]")
        # Official short/stacked representation: shared vocabulary, sum of
        # component lookups. Padding component 0 contributes a zero vector.
        embeds = self.backbone.embed_tokens(input_ids).sum(dim=-2)
        valid = attention_mask.to(device=embeds.device, dtype=torch.bool)
        # In Transformers 4.53.3, a prepared 4D mask bypasses create_causal_mask.
        # Explicit zeros above AND below the diagonal permit bidirectional SMTP.
        # This is only a padding mask, with no reference/graph attention rules.
        mask = embeds.new_zeros((batch, 1, 1, length))
        mask = mask.masked_fill(~valid[:, None, None, :], torch.finfo(embeds.dtype).min)
        mask = mask.expand(batch, 1, length, length)
        positions = torch.arange(length, device=embeds.device).unsqueeze(0)
        return self.backbone(inputs_embeds=embeds, attention_mask=mask,
                             position_ids=positions, use_cache=False, return_dict=True).last_hidden_state

    def forward(self, input_ids, labels=None, attention_mask=None):
        states = self.encode(input_ids, attention_mask)
        logits = self.lm_head(states)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape[:2]:
                raise ValueError("labels must have shape [batch, length], with -100 outside SMTP targets")
            valid = input_ids[:, :, 0] != 0
            if attention_mask is not None:
                valid = valid & attention_mask.to(device=valid.device, dtype=torch.bool)
            targets = labels.to(device=logits.device).masked_fill(~valid, -100)
            # No causal shift: SMTP predicts the identity at the masked position.
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                   targets.reshape(-1), ignore_index=-100)
        return GraphGPTHIROutput(states, logits, loss)

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return model.to(device)
