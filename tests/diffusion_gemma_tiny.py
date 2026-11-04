"""A tiny random DiffusionGemma checkpoint for CPU tests of the dInfer integration."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

PREFIX = "model.decoder."
LAYER_TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"]
TEXT_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 48,
    "moe_intermediate_size": 16,
    "num_hidden_layers": len(LAYER_TYPES),
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 8,
    "global_head_dim": 16,
    "num_experts": 8,
    "top_k_experts": 2,
    "vocab_size": 64,
    "sliding_window": 6,
    "rms_norm_eps": 1e-6,
    "final_logit_softcapping": 30.0,
    "layer_types": LAYER_TYPES,
    "rope_parameters": {
        "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    },
    "hidden_activation": "gelu_pytorch_tanh",
    "attention_bias": False,
}
CANVAS_LENGTH = 16


def _tensors(seed):
    c = TEXT_CONFIG
    g = torch.Generator().manual_seed(seed)
    H, I, I_moe, E = c["hidden_size"], c["intermediate_size"], c["moe_intermediate_size"], c["num_experts"]

    def rand(*shape, scale=0.2):
        return torch.randn(*shape, generator=g) * scale

    def norm(n):
        return 1.0 + rand(n, scale=0.05)

    t = {
        "embed_tokens.weight": rand(c["vocab_size"], H, scale=0.5),
        "norm.weight": norm(H),
        "self_conditioning.pre_norm.weight": norm(H),
        "self_conditioning.gate_proj.weight": rand(I, H),
        "self_conditioning.up_proj.weight": rand(I, H),
        "self_conditioning.down_proj.weight": rand(H, I),
    }
    for i, layer_type in enumerate(LAYER_TYPES):
        p = f"layers.{i}."
        sliding = layer_type == "sliding_attention"
        head_dim = c["head_dim"] if sliding else c["global_head_dim"]
        kv_heads = c["num_key_value_heads"] if sliding else c["num_global_key_value_heads"]
        q_out = c["num_attention_heads"] * head_dim
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                  "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"]:
            t[p + n + ".weight"] = norm(H)
        t[p + "layer_scalar"] = torch.tensor([0.9])
        t[p + "self_attn.q_proj.weight"] = rand(q_out, H)
        t[p + "self_attn.k_proj.weight"] = rand(kv_heads * head_dim, H)
        if sliding:
            t[p + "self_attn.v_proj.weight"] = rand(kv_heads * head_dim, H)
        t[p + "self_attn.o_proj.weight"] = rand(H, q_out)
        t[p + "self_attn.q_norm.weight"] = norm(head_dim)
        t[p + "self_attn.k_norm.weight"] = norm(head_dim)
        t[p + "mlp.gate_proj.weight"] = rand(I, H)
        t[p + "mlp.up_proj.weight"] = rand(I, H)
        t[p + "mlp.down_proj.weight"] = rand(H, I)
        t[p + "router.proj.weight"] = rand(E, H, scale=1.0)
        t[p + "router.scale"] = norm(H)
        t[p + "router.per_expert_scale"] = norm(E)
        t[p + "experts.gate_up_proj"] = rand(E, 2 * I_moe, H)
        t[p + "experts.down_proj"] = rand(E, H, I_moe)
    return {PREFIX + k: v.to(torch.bfloat16).contiguous() for k, v in t.items()}


def write_tiny_checkpoint(directory, seed=0):
    """Write config.json, model.safetensors and its index to `directory`; returns the path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = _tensors(seed)
    save_file(tensors, str(directory / "model.safetensors"))
    index = {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    (directory / "config.json").write_text(json.dumps({"text_config": TEXT_CONFIG, "canvas_length": CANVAS_LENGTH}))
    return str(directory)


def load_tiny_model(directory, seed=0):
    from dinfer.model import DiffusionGemmaModelLM

    path = write_tiny_checkpoint(directory, seed)
    return DiffusionGemmaModelLM.from_pretrained(path, device="cpu", rank=0, world_size=1).eval()
