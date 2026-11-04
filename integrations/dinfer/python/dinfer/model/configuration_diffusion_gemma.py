"""Configuration for DiffusionGemma (google/diffusiongemma-26B-A4B-it), text path only.

Read directly from the checkpoint's config.json ("text_config" + "canvas_length"), so transformers 5
(which defines the official DiffusionGemma config class) is not required.
"""
import json
import os
from dataclasses import dataclass

import torch


@dataclass
class DiffusionGemmaConfig:
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    num_global_key_value_heads: int
    head_dim: int
    global_head_dim: int
    num_experts: int
    top_k_experts: int
    vocab_size: int
    sliding_window: int
    rms_norm_eps: float
    final_logit_softcapping: float
    layer_types: list
    rope_parameters: dict
    canvas_length: int

    @classmethod
    def from_pretrained(cls, path):
        cfg = json.load(open(os.path.join(path, "config.json")))
        text = cfg["text_config"]
        assert text["hidden_activation"] == "gelu_pytorch_tanh" and not text["attention_bias"]
        fields = {k: text[k] for k in cls.__dataclass_fields__ if k != "canvas_length"}
        return cls(canvas_length=cfg["canvas_length"], **fields)

    @property
    def embed_scale(self):
        # HF stores sqrt(hidden_size) as a buffer and casts it to the bf16 weight dtype: 53.066 -> 53.0
        return torch.tensor(self.hidden_size ** 0.5, dtype=torch.bfloat16).item()
