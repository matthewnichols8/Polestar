"""Fuse rotary embedding reads, rotation, and dtype conversion."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _rotary_kernel(
        x_ptr, sin_ptr, cos_ptr, positions_ptr, out_ptr,
        heads: tl.constexpr, tokens, dim: tl.constexpr,
        stride_batch, stride_head, stride_token, stride_dim,
        position_start, indexed_positions: tl.constexpr,
        full_precision: tl.constexpr, block: tl.constexpr,
    ):
        batch_head = tl.program_id(0)
        token = tl.program_id(1)
        head = batch_head % heads
        batch = batch_head // heads
        channels = tl.arange(0, block)
        half = dim // 2
        valid = channels < dim
        rotated_channels = tl.where(channels < half, channels + half, channels - half)
        x_base = x_ptr + batch * stride_batch + head * stride_head + token * stride_token
        values = tl.load(x_base + channels * stride_dim, valid, other=0)
        other = tl.load(x_base + rotated_channels * stride_dim, valid, other=0)
        rotated = tl.where(channels < half, -other.to(tl.float32), other.to(tl.float32))
        if indexed_positions:
            position = tl.load(positions_ptr + token)
        else:
            position = position_start + token
        sinusoid = tl.load(sin_ptr + position * dim + channels, valid, other=0)
        cosinusoid = tl.load(cos_ptr + position * dim + channels, valid, other=0)
        if full_precision:
            left = values.to(tl.float32) * cosinusoid.to(tl.float32)
            right = rotated * sinusoid.to(tl.float32)
            result = (left + right).to(values.dtype)
        else:
            left = (values.to(tl.float32) * cosinusoid.to(values.dtype).to(tl.float32)).to(values.dtype)
            right = (rotated * sinusoid.to(values.dtype).to(tl.float32)).to(values.dtype)
            result = (left.to(tl.float32) + right.to(tl.float32)).to(values.dtype)
        tl.store(out_ptr + (batch_head * tokens + token) * dim + channels, result, valid)


def apply_rotary(
    x: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    *,
    position_start: int = 0,
    positions: torch.Tensor | None = None,
    full_precision: bool = True,
) -> torch.Tensor:
    """Apply RoPE to [batch, head, token, channel] without materializing FP32 keys."""
    if triton is None or not x.is_cuda:
        raise RuntimeError("the fused RoPE path requires Triton and CUDA")
    if x.ndim != 4 or x.shape[-1] % 2:
        raise ValueError("RoPE expects an even head dimension and four-dimensional input")
    batch, heads, tokens, dim = x.shape
    output = torch.empty((batch, heads, tokens, dim), dtype=x.dtype, device=x.device)
    _rotary_kernel[(batch * heads, tokens)](
        x, sin.contiguous(), cos.contiguous(),
        positions if positions is not None else x, output,
        heads, tokens, dim, *x.stride(), position_start,
        positions is not None, full_precision,
        block=triton.next_power_of_2(dim), num_warps=4,
        enable_fp_fusion=False,
    )
    return output
