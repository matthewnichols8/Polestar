"""DiffusionGemma (google/diffusiongemma-26B-A4B-it), text path, for dInfer.

A re-implementation of HF transformers' `DiffusionGemmaForBlockDiffusion` text model
(transformers/models/diffusion_gemma/modeling_diffusion_gemma.py), without a transformers-5 dependency.
Verified layer by layer against the HF implementation.

Unlike LLaDA, DiffusionGemma is a *uniform-noise* diffusion model: the canvas starts as random tokens
(no mask token) and is denoised by an entropy-bound sampler. The model has one set of weights used in two modes:
  * encode(ids)    - causal pass over the context; appends K/V to the cache (HF "encoder").
  * denoise(canvas) - bidirectional pass over a canvas that reads, but never writes, the cache;
                      returns softcapped logits (HF "decoder" + lm_head).

Weights and precision (Turing GPUs have fp16 tensor cores but no bf16 ones):
  * the checkpoint's bf16 weights are used unquantized and held in fp16 for cuBLAS (bf16 -> fp16 is exact
    except for values below fp16's normal range, ~6e-5 in magnitude);
  * big matmuls run in fp16 with fp32 outputs; residual stream, RMSNorms, RoPE, router and attention run in fp32;
  * MoE experts are split across ranks (expert parallel). With expert_dtype='bf16' (default, same weights as the
    HF model) the experts of as many layers as fit stay on the GPU and the rest are kept in pinned CPU memory and
    streamed to the GPU every pass (the model is ~50 GB; 2x24 GB GPUs cannot hold it). With expert_dtype='int8'
    they are quantized per output channel instead, which fits on the GPUs (faster, slightly different weights).
"""
import json
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from ..decoding.utils import PerLayerKVCache
from .configuration_diffusion_gemma import DiffusionGemmaConfig

FP16_MAX = 65504.0
# Set DGEMMA_CHECK_OVERFLOW=1 to assert that every fp16 matmul output is finite (slow; for debugging).
CHECK_OVERFLOW = os.environ.get("DGEMMA_CHECK_OVERFLOW", "0") == "1"


# ----------------------------------------------------------------------------- helpers

def rmsnorm(x, weight=None, eps=1e-6):
    """Gemma4 RMSNorm in fp32. NOTE: the weight multiplies directly (no `1 + weight` offset)."""
    x = x.float()
    out = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
    return out if weight is None else out * weight


def linear16(x, weight):
    """fp16 cuBLAS matmul with an fp32 result. `weight` is stored fp16 (computed in fp32 on CPU)."""
    if weight.device.type == "cpu":
        return F.linear(x.float(), weight.float())
    out = F.linear(x.to(torch.float16), weight)
    if CHECK_OVERFLOW and not torch.isfinite(out).all():
        raise FloatingPointError(f"fp16 overflow in linear {tuple(weight.shape)}")
    return out.float()


def gelu_tanh(x):
    return F.gelu(x, approximate="tanh")


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_inv_freq(cfg, layer_type):
    """Inverse frequencies matching HF's default and 'proportional' rope types."""
    p = cfg.rope_parameters[layer_type]
    head_dim = cfg.global_head_dim if layer_type == "full_attention" else cfg.head_dim
    base = p["rope_theta"]
    if p["rope_type"] == "default":
        dim = int(head_dim * p.get("partial_rotary_factor", 1.0))
        return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    assert p["rope_type"] == "proportional", p
    # Only the first `rope_angles` frequency pairs rotate, with exponents over the FULL head_dim;
    # the rest get frequency 0 (cos=1, sin=0). rotate_half still spans all of head_dim.
    rope_angles = int(p.get("partial_rotary_factor", 1.0) * head_dim // 2)
    rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float) / head_dim))
    inv = torch.cat([rotated, torch.zeros(head_dim // 2 - rope_angles)])
    return inv / p.get("factor", 1.0)


def rope_cos_sin(inv_freq, positions):
    """positions: [L] long -> cos, sin: [L, head_dim] fp32."""
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x: [B, L, H, D]; cos/sin: [L, D]."""
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return x * cos + rotate_half(x) * sin


class Holder(nn.Module):
    """Plain container whose tensors are registered as buffers (inference only, no autograd)."""


def _set_tensor(root, dotted, tensor):
    *path, leaf = dotted.split(".")
    mod = root
    for p in path:
        mod = getattr(mod, p)
    mod.register_buffer(leaf, tensor)


# ----------------------------------------------------------------------------- modules

class Attention(Holder):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = cfg.head_dim if self.is_sliding else cfg.global_head_dim
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads if self.is_sliding else cfg.num_global_key_value_heads
        self.window = cfg.sliding_window
        self.eps = cfg.rms_norm_eps
        names = ["q_proj", "k_proj", "o_proj", "q_norm", "k_norm"] + (["v_proj"] if self.is_sliding else [])
        for n in names:
            self.add_module(n, Holder())

    def forward(self, x, cos, sin, cache, mode, q_pos):
        """x: [B, L, H] fp32 (already input-normed). mode: 'encode' (causal, writes cache) or 'decode'."""
        B, L, _ = x.shape
        q = linear16(x, self.q_proj.weight).view(B, L, self.num_heads, self.head_dim)
        k_raw = linear16(x, self.k_proj.weight).view(B, L, self.num_kv_heads, self.head_dim)
        v = linear16(x, self.v_proj.weight).view(B, L, self.num_kv_heads, self.head_dim) if self.is_sliding else k_raw

        q = apply_rope(rmsnorm(q, self.q_norm.weight, self.eps), cos, sin).transpose(1, 2)
        k = apply_rope(rmsnorm(k_raw, self.k_norm.weight, self.eps), cos, sin).transpose(1, 2)
        v = rmsnorm(v, None, self.eps).transpose(1, 2)            # v_norm has no weight; K==V source on full layers

        ck, cv = cache.get(self.layer_idx)
        keys = k if ck is None else torch.cat([ck, k], dim=2)
        vals = v if cv is None else torch.cat([cv, v], dim=2)

        if mode == "encode":
            cached_pos = cache.key_positions(self.layer_type)
            k_pos = q_pos if cached_pos is None else torch.cat([cached_pos, q_pos])
            mask = k_pos[None, :] <= q_pos[:, None]
            if self.is_sliding:
                mask &= k_pos[None, :] > q_pos[:, None] - self.window
            cache.append(self.layer_idx, self.layer_type, k, v)
        else:
            mask = None  # canvas attends bidirectionally to the cached prefix and itself

        out = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask, scale=1.0, enable_gqa=True)
        return linear16(out.transpose(1, 2).reshape(B, L, -1), self.o_proj.weight)

    # ------------------------------------------------------------------ primitives for cross-step canvas caches

    def keys_values(self, x, cos, sin, cache):
        """Decode-mode K/V for every canvas token, prepended with the prefix cache: [B, H_kv, S, D] each."""
        B, L, _ = x.shape
        k_raw = linear16(x, self.k_proj.weight).view(B, L, self.num_kv_heads, self.head_dim)
        v = linear16(x, self.v_proj.weight).view(B, L, self.num_kv_heads, self.head_dim) if self.is_sliding else k_raw
        k = apply_rope(rmsnorm(k_raw, self.k_norm.weight, self.eps), cos, sin).transpose(1, 2)
        v = rmsnorm(v, None, self.eps).transpose(1, 2)
        ck, cv = cache.get(self.layer_idx)
        keys = k if ck is None else torch.cat([ck, k], dim=2)
        vals = v if cv is None else torch.cat([cv, v], dim=2)
        return keys, vals

    def attend(self, x_sel, cos_sel, sin_sel, keys, vals):
        """Bidirectional attention for a subset of canvas queries over all keys. x_sel: [B, T, H] (input-normed)."""
        B, T, _ = x_sel.shape
        q = linear16(x_sel, self.q_proj.weight).view(B, T, self.num_heads, self.head_dim)
        q = apply_rope(rmsnorm(q, self.q_norm.weight, self.eps), cos_sel, sin_sel).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, keys, vals, scale=1.0, enable_gqa=True)
        return linear16(out.transpose(1, 2).reshape(B, T, -1), self.o_proj.weight)

    def proxy_attention(self, x_c, keys):
        """Attention probabilities of position-free queries (e.g. cluster centroids; input-normed, [K, H]) over
        `keys` [1, H_kv, S, D], without RoPE. Returns softmax probabilities [heads, K, S] fp32."""
        K = x_c.shape[0]
        q = linear16(x_c, self.q_proj.weight).view(K, self.num_heads, self.head_dim)
        q = rmsnorm(q, self.q_norm.weight, self.eps).transpose(0, 1)                       # [heads, K, D]
        k = keys[0].float().repeat_interleave(self.num_heads // self.num_kv_heads, dim=0)  # [heads, S, D]
        return torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1)


class Experts(Holder):
    """This rank's slice [e0, e1) of one layer's experts.

    Storage `placement`:
      'gpu'  - fp16 on the GPU (the checkpoint's bf16 values);
      'cpu'  - fp16 in pinned CPU memory, copied into the shared fp16 scratch on every forward;
      'int8' - int8 on the GPU with per-output-channel scales, dequantized into the shared scratch.
    Forward: group tokens by expert into padded batches, two cuBLAS bmm's with tanh-GELU, weighted scatter-add.
    Returns this rank's partial sum (fp32); the caller all-reduces across ranks.
    """

    def __init__(self, cfg, e0, e1, scratch):
        super().__init__()
        self.e0, self.e1, self.n_loc = e0, e1, e1 - e0
        self.inter = cfg.moe_intermediate_size
        self.scratch = scratch  # dict with 'w13' [n_loc, 2I, H] and 'w2' [n_loc, H, I] fp16, shared by all layers
        self.placement = None

    def load(self, gate_up, down, device, placement):
        """gate_up: [n_loc, 2I, H], down: [n_loc, H, I] (bf16, this rank's experts only)."""
        self.placement = placement
        for name, w in (("w13", gate_up), ("w2", down)):
            if placement == "gpu":
                self.register_buffer(name, w.to(device).to(torch.float16))
            elif placement == "cpu":
                setattr(self, f"{name}_cpu", w.to(torch.float16).pin_memory())   # not a buffer: stays on the CPU
            elif placement == "int8":
                w = w.to(device).float()
                scale = w.abs().amax(dim=-1).clamp(min=1e-8) / 127.0
                q = torch.round(w / scale[..., None]).clamp(-127, 127).to(torch.int8)
                self.register_buffer(f"{name}_q", q)
                self.register_buffer(f"{name}_s", scale.to(torch.float16)[..., None])
                del w
            else:
                raise ValueError(placement)

    def _weights(self):
        """fp16 [n_loc, 2I, H] and [n_loc, H, I] on the GPU."""
        if self.placement == "gpu":
            return self.w13, self.w2
        if self.placement == "cpu":
            return (self.scratch["w13"].copy_(self.w13_cpu, non_blocking=True),
                    self.scratch["w2"].copy_(self.w2_cpu, non_blocking=True))
        return (torch.mul(self.w13_q, self.w13_s, out=self.scratch["w13"]),
                torch.mul(self.w2_q, self.w2_s, out=self.scratch["w2"]))

    sparse_dequant = False  # set by cross-step canvas caches when only a subset of tokens is recomputed

    def _weights_subset(self, used):
        """Like _weights, for the local experts `used` only (placements 'int8' and 'gpu')."""
        n = used.numel()
        if self.placement == "gpu":
            return self.w13[used], self.w2[used]
        return (torch.mul(self.w13_q[used], self.w13_s[used], out=self.scratch["w13"][:n]),
                torch.mul(self.w2_q[used], self.w2_s[used], out=self.scratch["w2"][:n]))

    def forward(self, y, topk_w, topk_ids):
        """y: [T, H] fp32 (pre_feedforward_layernorm_2 output); topk_w fp32 / topk_ids long: [T, K]."""
        T, H = y.shape
        out = torch.zeros(T, H, device=y.device, dtype=torch.float32)
        K = topk_ids.shape[1]
        flat = topk_ids.flatten() - self.e0
        keep = (flat >= 0) & (flat < self.n_loc)
        if not bool(keep.any()):
            return out
        tok = torch.arange(T, device=y.device).repeat_interleave(K)[keep]
        wf, flat = topk_w.flatten()[keep], flat[keep]
        order = flat.argsort(stable=True)
        flat, wf, tok = flat[order], wf[order], tok[order]
        slot = torch.arange(flat.numel(), device=y.device) - torch.searchsorted(
            flat, torch.arange(self.n_loc, device=y.device))[flat]
        cap = int(slot.max().item()) + 1

        n_groups = self.n_loc
        if self.sparse_dequant and self.placement in ("int8", "gpu") and \
                (used := torch.unique(flat)).numel() < self.n_loc:
            # Partial recompute: materialize only the experts these tokens route to. Relabeling the
            # (sorted) expert ids 0..n-1 keeps the order, so `slot` is unchanged.
            remap = torch.empty(self.n_loc, dtype=flat.dtype, device=y.device)
            remap[used] = torch.arange(used.numel(), device=y.device)
            flat, n_groups = remap[flat], used.numel()
            w13, w2 = self._weights_subset(used)
        else:
            w13, w2 = self._weights()
        X = torch.zeros(n_groups, cap, H, device=y.device, dtype=torch.float16)
        X[flat, slot] = y[tok].to(torch.float16)
        gu = torch.bmm(X, w13.transpose(1, 2))
        act = gelu_tanh(gu[..., :self.inter].float()) * gu[..., self.inter:].float()
        o = torch.bmm(act.to(torch.float16), w2.transpose(1, 2))
        if CHECK_OVERFLOW and not (torch.isfinite(gu).all() and torch.isfinite(o).all()):
            raise FloatingPointError("fp16 overflow in experts")
        return out.index_add_(0, tok, o[flat, slot].float() * wf[:, None])


class Layer(Holder):
    def __init__(self, cfg, layer_idx, e0, e1, scratch):
        super().__init__()
        self.config = cfg
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.eps = cfg.rms_norm_eps
        self.self_attn = Attention(cfg, layer_idx)
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                  "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2", "mlp", "router"]:
            self.add_module(n, Holder())
        for n in ["gate_proj", "up_proj", "down_proj"]:
            self.mlp.add_module(n, Holder())
        self.router.add_module("proj", Holder())
        self.experts = Experts(cfg, e0, e1, scratch)

    def forward(self, h, cos, sin, cache, mode, q_pos, attn_only=False):
        B, L, H = h.shape
        r = h
        a = self.self_attn(rmsnorm(h, self.input_layernorm.weight, self.eps), cos, sin, cache, mode, q_pos)
        h = r + rmsnorm(a, self.post_attention_layernorm.weight, self.eps)
        if attn_only:  # encode of the last layer: only its K/V matter
            return h
        return self._ffn(h)

    def _ffn(self, h):
        """Dense MLP + MoE half of the layer (incl. the cross-rank all-reduce). h: [B, L, H] -> layer output."""
        B, L, H = h.shape
        r = h
        # dense MLP branch
        x1 = rmsnorm(h, self.pre_feedforward_layernorm.weight, self.eps)
        g, u = linear16(x1, self.mlp.gate_proj.weight), linear16(x1, self.mlp.up_proj.weight)
        h1 = rmsnorm(linear16(gelu_tanh(g) * u, self.mlp.down_proj.weight), self.post_feedforward_layernorm_1.weight, self.eps)

        # MoE branch: router reads the un-normed residual
        flat = r.reshape(-1, H)
        xr = rmsnorm(flat, None, self.eps) * self.router.scale * H ** -0.5
        probs = torch.softmax(xr @ self.router.proj.weight.T, dim=-1)
        topk_w, topk_ids = probs.topk(self.config.top_k_experts, dim=-1)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True) * self.router.per_expert_scale[topk_ids]
        y = rmsnorm(flat, self.pre_feedforward_layernorm_2.weight, self.eps)
        h2 = self.experts(y, topk_w, topk_ids)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(h2)
        h2 = rmsnorm(h2.view(B, L, H), self.post_feedforward_layernorm_2.weight, self.eps)

        h = r + rmsnorm(h1 + h2, self.post_feedforward_layernorm.weight, self.eps)
        return h * self.layer_scalar

    # ------------------------------------------------------------------ primitives for cross-step canvas caches
    def input_norm(self, h):
        return rmsnorm(h, self.input_layernorm.weight, self.eps)

    def post_attention(self, h, a):
        """Residual + post-attention norm: the layer state after attention, before `_ffn`."""
        return h + rmsnorm(a, self.post_attention_layernorm.weight, self.eps)

    def proxy_attention(self, hidden, keys):
        """Position-free queries from layer-input hidden states ([K, H], e.g. cluster centroids) ->
        attention probabilities [heads, K, S] over `keys`."""
        return self.self_attn.proxy_attention(self.input_norm(hidden), keys)


class DiffusionGemmaModelLM(Holder):
    """DiffusionGemma with experts split across ranks (expert parallel over the torch.distributed world).

    By default the rank/world size come from torch.distributed (initialized directly, or through
    vllm.distributed as in dInfer's other runners); pass them explicitly to override.
    """
    PREFIX = "model.decoder."

    def __init__(self, config, device=None, rank=None, world_size=None, expert_dtype="bf16", memory_reserve_gib=3.5):
        """expert_dtype: 'bf16' (checkpoint weights, CPU offload as needed) or 'int8' (quantized, all on GPU).
        memory_reserve_gib: GPU memory left free for activations when deciding how many layers' experts fit."""
        super().__init__()
        assert expert_dtype in ("bf16", "int8"), expert_dtype
        self.expert_dtype, self.memory_reserve_gib = expert_dtype, memory_reserve_gib
        cfg = config
        if rank is None or world_size is None:
            initialized = dist.is_initialized()
            rank = dist.get_rank() if initialized else 0
            world_size = dist.get_world_size() if initialized else 1
        device = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        assert cfg.num_experts % world_size == 0
        self.config, self.rank, self.world_size, self.device = cfg, rank, world_size, device
        n_loc = cfg.num_experts // world_size
        self.e0, self.e1 = rank * n_loc, (rank + 1) * n_loc
        self.scratch = {}  # fp16 staging for CPU-resident or int8 experts; allocated in load_weights if needed
        self.layers = nn.ModuleList([Layer(cfg, i, self.e0, self.e1, self.scratch) for i in range(cfg.num_hidden_layers)])
        for n in ["embed_tokens", "norm", "self_conditioning"]:
            self.add_module(n, Holder())
        for n in ["pre_norm", "gate_proj", "up_proj", "down_proj"]:
            self.self_conditioning.add_module(n, Holder())
        self.inv_freq = {lt: rope_inv_freq(cfg, lt).to(self.device) for lt in set(cfg.layer_types)}

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, model_path, device=None, rank=None, world_size=None, expert_dtype="bf16",
                        memory_reserve_gib=3.5):
        model = cls(DiffusionGemmaConfig.from_pretrained(model_path), device, rank, world_size, expert_dtype,
                    memory_reserve_gib)
        model.load_weights(model_path)
        return model

    def _target_dtype(self, name):
        if name.endswith(("proj.weight", "embed_tokens.weight")) and "router.proj" not in name:
            return torch.float16  # matmul weights -> fp16 cuBLAS
        return torch.float32       # norms, scalars, router

    @torch.no_grad()
    def load_weights(self, path):
        """Stream tensors one by one (never a whole shard in host RAM).

        Pass 1 loads the non-expert weights to the GPU. Pass 2 loads this rank's experts layer by layer; with
        expert_dtype='bf16', as many layers as fit in the GPU's currently free memory (minus the reserve) stay
        on the GPU and the rest go to pinned CPU memory.
        """
        cfg = self.config
        weight_map = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        expected = {n for n, _ in self._expected_tensors()}
        is_expert = lambda name: name.endswith(("experts.gate_up_proj", "experts.down_proj"))

        # pass 1: everything except the experts
        by_shard = defaultdict(list)
        for key, shard in weight_map.items():
            if key.startswith(self.PREFIX) and not is_expert(key[len(self.PREFIX):]):
                by_shard[shard].append(key)
        loaded = set()
        for shard, keys in sorted(by_shard.items()):
            with safe_open(os.path.join(path, shard), framework="pt", device="cpu") as f:
                for key in keys:
                    name = key[len(self.PREFIX):]
                    if name not in expected:
                        raise KeyError(f"unexpected checkpoint tensor {key}")
                    t = f.get_tensor(key).to(self.device)
                    dtype = self._target_dtype(name)
                    if dtype == torch.float16 and t.abs().max().item() > FP16_MAX:
                        raise ValueError(f"{key} exceeds fp16 range")
                    _set_tensor(self, name, t.to(dtype))
                    loaded.add(name)

        # decide where each layer's experts live
        L, n_loc = cfg.num_hidden_layers, self.e1 - self.e0
        H, I = cfg.hidden_size, cfg.moe_intermediate_size
        layer_bytes = n_loc * (2 * I * H + H * I) * 2   # fp16 / bf16
        if self.expert_dtype == "int8":
            placements = ["int8"] * L
        elif self.device.type != "cuda":
            placements = ["gpu"] * L   # resident on the (CPU) device
        else:
            torch.cuda.synchronize(self.device)
            free = torch.cuda.mem_get_info(self.device)[0] - self.memory_reserve_gib * 2 ** 30
            n_gpu = min(L, max(0, int(free // layer_bytes)))
            if n_gpu < L:  # offloading needs one layer of scratch
                n_gpu = max(0, int((free - layer_bytes) // layer_bytes))
            placements = ["gpu"] * n_gpu + ["cpu"] * (L - n_gpu)
        if any(p != "gpu" for p in placements):
            self.scratch["w13"] = torch.empty(n_loc, 2 * I, H, device=self.device, dtype=torch.float16)
            self.scratch["w2"] = torch.empty(n_loc, H, I, device=self.device, dtype=torch.float16)
        self.expert_placement = placements

        # pass 2: this rank's experts, layer by layer
        for i, placement in enumerate(placements):
            ws = {}
            for part in ("gate_up_proj", "down_proj"):
                key = f"{self.PREFIX}layers.{i}.experts.{part}"
                with safe_open(os.path.join(path, weight_map[key]), framework="pt", device="cpu") as f:
                    ws[part] = f.get_slice(key)[self.e0:self.e1]   # read only this rank's experts
                loaded.add(key[len(self.PREFIX):])
            self.layers[i].experts.load(ws["gate_up_proj"], ws["down_proj"], self.device, placement)
            del ws

        missing = expected - loaded
        if missing:
            raise KeyError(f"missing tensors: {sorted(missing)[:10]} ... ({len(missing)})")

    def placement_summary(self):
        p = self.expert_placement
        where = "GPU" if self.device.type == "cuda" else str(self.device)
        if self.expert_dtype == "int8":
            return f"experts {self.e0}-{self.e1 - 1}: int8, all {len(p)} layers on {where}"
        n_cpu = p.count("cpu")
        return (f"experts {self.e0}-{self.e1 - 1}: bf16 weights, {len(p) - n_cpu} layers on {where}, "
                f"{n_cpu} in pinned CPU memory" + (f" (layers {p.index('cpu')}-{len(p) - 1})" if n_cpu else ""))

    def _expected_tensors(self):
        yield "embed_tokens.weight", None
        yield "norm.weight", None
        for n in ["pre_norm", "gate_proj", "up_proj", "down_proj"]:
            yield f"self_conditioning.{n}.weight", None
        for i, layer in enumerate(self.layers):
            p = f"layers.{i}."
            for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                      "post_feedforward_layernorm", "post_feedforward_layernorm_1",
                      "post_feedforward_layernorm_2", "pre_feedforward_layernorm_2"]:
                yield p + n + ".weight", None
            yield p + "layer_scalar", None
            for n in ["q_proj", "k_proj", "o_proj", "q_norm", "k_norm"] + (["v_proj"] if layer.self_attn.is_sliding else []):
                yield p + f"self_attn.{n}.weight", None
            for n in ["gate_proj", "up_proj", "down_proj"]:
                yield p + f"mlp.{n}.weight", None
            for n in ["proj.weight", "scale", "per_expert_scale"]:
                yield p + f"router.{n}", None
            yield p + "experts.gate_up_proj", None
            yield p + "experts.down_proj", None

    # ------------------------------------------------------------------ forward
    def new_cache(self):
        return PerLayerKVCache(self.config.num_hidden_layers, self.config.layer_types, self.config.sliding_window)

    def _rope(self, positions):
        return {lt: rope_cos_sin(f, positions) for lt, f in self.inv_freq.items()}

    def _embed(self, ids):
        return self.embed_tokens.weight[ids].float() * self.config.embed_scale

    @torch.no_grad()
    def encode(self, input_ids, cache, chunk_size=256):
        """Causal pass over `input_ids` [B, L]; appends their K/V to `cache` at the next positions.

        Long inputs are encoded in chunks of `chunk_size` tokens so activation memory does not grow with the
        prompt length. This is exact: each chunk attends causally to the cache plus itself, and sliding-window
        layers keep the last `sliding_window - 1` tokens, which is everything a following chunk can see."""
        for c in range(0, input_ids.shape[1], chunk_size):
            ids = input_ids[:, c:c + chunk_size]
            start = cache.seq_length()
            pos = torch.arange(start, start + ids.shape[1], device=self.device)
            rope = self._rope(pos)
            h = self._embed(ids)
            last = len(self.layers) - 1
            for i, layer in enumerate(self.layers):
                cos, sin = rope[layer.layer_type]
                h = layer(h, cos, sin, cache, "encode", pos, attn_only=(i == last))
            cache.append_positions(pos)

    @torch.no_grad()
    def denoise(self, canvas, cache, self_conditioning_logits=None, layer_runner=None):
        """Bidirectional pass over `canvas` [B, C] placed right after the cache.
        Returns softcapped fp32 logits [B, C, V]. Does not modify `cache`.

        layer_runner: optional cross-step canvas cache. It gets `begin_step(canvas)`, then runs every layer as
        `run_layer(layer, h, cos, sin, cache, positions)` (and may recompute only some tokens), then `end_step()`.
        None runs every layer normally."""
        if layer_runner is not None:
            layer_runner.begin_step(canvas)
        start = cache.seq_length()
        pos = torch.arange(start, start + canvas.shape[1], device=self.device)
        rope = self._rope(pos)
        emb = self._embed(canvas)
        sc = self.self_conditioning
        if self_conditioning_logits is None:
            soft = torch.zeros_like(emb)
        else:
            probs = torch.softmax(self_conditioning_logits.float(), dim=-1).to(torch.float16)
            soft = torch.matmul(probs, self.embed_tokens.weight).float() * self.config.embed_scale
        n = rmsnorm(soft, sc.pre_norm.weight, self.config.rms_norm_eps)
        s = linear16(gelu_tanh(linear16(n, sc.gate_proj.weight)) * linear16(n, sc.up_proj.weight), sc.down_proj.weight)
        h = rmsnorm(emb + s, None, self.config.rms_norm_eps)
        for layer in self.layers:
            cos, sin = rope[layer.layer_type]
            if layer_runner is None:
                h = layer(h, cos, sin, cache, "decode", pos)
            else:
                h = layer_runner.run_layer(layer, h, cos, sin, cache, pos)
        if layer_runner is not None:
            layer_runner.end_step()
        h = rmsnorm(h, self.norm.weight, self.config.rms_norm_eps)
        logits = linear16(h, self.embed_tokens.weight)  # tied lm_head
        cap = self.config.final_logit_softcapping
        return torch.tanh(logits / cap) * cap
