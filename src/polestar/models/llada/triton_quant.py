"""Fused decoding of packed NVFP4 residuals and cluster references."""

from __future__ import annotations

import torch

from .system_clustering import dequantize_residual_codes, unpack_codes

try:
    import triton
    import triton.language as tl
except ImportError:  # Triton is optional for CPU-only installation.
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _decode_nvfp4_kernel(
        packed_ptr, scale_ptr, reference_ptr, output_ptr,
        dim: tl.constexpr, groups: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        channels = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = channels < dim
        payload = tl.load(packed_ptr + row * (dim // 2) + channels // 2, mask, other=0)
        code = tl.where(channels % 2 == 0, payload & 15, payload >> 4)
        mag = code & 7
        value = tl.where(mag == 0, 0.0,
                tl.where(mag == 1, 0.5,
                tl.where(mag == 2, 1.0,
                tl.where(mag == 3, 1.5,
                tl.where(mag == 4, 2.0,
                tl.where(mag == 5, 3.0,
                tl.where(mag == 6, 4.0, 6.0)))))))
        value = tl.where((code & 8) != 0, -value, value)
        scale_bits = tl.load(scale_ptr + row * groups + channels // 16, mask, other=0).to(tl.int32)
        exponent = (scale_bits >> 3) & 15
        mantissa = scale_bits & 7
        scale = tl.where(
            exponent == 0,
            mantissa.to(tl.float32) * (2.0 ** -9),
            (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0),
        )
        reference = tl.load(reference_ptr + row * dim + channels, mask, other=0).to(tl.float32)
        tl.store(output_ptr + row * dim + channels, reference + value * scale, mask)


def decode_packed_residuals(
    packed: torch.Tensor,
    scales: torch.Tensor,
    reference: torch.Tensor,
    *,
    precision: str,
    group_size: int,
    use_triton: bool = True,
) -> torch.Tensor:
    """Restore selected hidden-state rows, adding their block-entry centroids."""
    if (
        precision == "fp4"
        and group_size == 16
        and use_triton
        and triton is not None
        and packed.is_cuda
    ):
        rows = packed.shape[0] * packed.shape[1]
        dim = reference.shape[-1]
        result = torch.empty_like(reference)
        _decode_nvfp4_kernel[(rows, triton.cdiv(dim, 256))](
            packed.contiguous(), scales.view(torch.uint8).contiguous(), reference.contiguous(), result,
            dim, dim // 16, BLOCK=256,
        )
        return result
    residual = dequantize_residual_codes(
        unpack_codes(packed), scales, precision=precision, group_size=group_size
    )
    return (reference.float() + residual).to(reference.dtype)
