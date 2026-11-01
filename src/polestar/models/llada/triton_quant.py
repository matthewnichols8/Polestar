"""Fused encoding and decoding of NVFP4 centroid residuals."""

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
    def _encode_nvfp4_kernel(
        hidden_ptr, centers_ptr, labels_ptr, packed_ptr, scales_ptr,
        width: tl.constexpr, dim: tl.constexpr, total_length: tl.constexpr,
        centers_per_layer: tl.constexpr, start: tl.constexpr,
        groups_per_program: tl.constexpr,
    ):
        row = tl.program_id(0)
        groups = tl.program_id(1) * groups_per_program + tl.arange(0, groups_per_program)
        valid = groups < dim // 16
        batch = row // width
        local = row % width
        channels = groups[:, None] * 16 + tl.arange(0, 16)[None, :]
        label = tl.maximum(tl.load(labels_ptr + batch * total_length + start + local), 0)
        hidden = tl.load(
            hidden_ptr + row * dim + channels, mask=valid[:, None], other=0
        ).to(tl.float32)
        reference = tl.load(
            centers_ptr + (batch * centers_per_layer + label) * dim + channels,
            mask=valid[:, None], other=0,
        ).to(tl.float32)
        residual = hidden - reference
        scale = tl.maximum(tl.max(tl.abs(residual), 1) / 6.0, 2.0 ** -9)
        scale = tl.minimum(scale, 448.0).to(tl.float8e4nv)
        scale_bits = scale.to(tl.uint8, bitcast=True)
        normalized = tl.div_rn(tl.abs(residual), scale.to(tl.float32)[:, None])

        best_code = tl.full((groups_per_program, 16), 0, tl.uint8)
        best_error = normalized
        for code in tl.static_range(1, 8):
            if code == 1:
                level = 0.5
            elif code == 2:
                level = 1.0
            elif code == 3:
                level = 1.5
            elif code == 4:
                level = 2.0
            elif code == 5:
                level = 3.0
            elif code == 6:
                level = 4.0
            else:
                level = 6.0
            error = tl.abs(normalized - level)
            closer = error < best_error
            best_code = tl.where(closer, code, best_code)
            best_error = tl.minimum(best_error, error)
        signed_code = best_code | tl.where(residual < 0, 8, 0).to(tl.uint8)
        halves = tl.arange(0, 8)
        low = tl.gather(
            signed_code,
            tl.broadcast_to((halves * 2)[None, :], (groups_per_program, 8)),
            axis=1,
        )
        high = tl.gather(
            signed_code,
            tl.broadcast_to((halves * 2 + 1)[None, :], (groups_per_program, 8)),
            axis=1,
        )
        tl.store(
            packed_ptr + row * (dim // 2) + groups[:, None] * 8 + halves[None, :],
            low | (high << 4),
            mask=valid[:, None],
        )
        tl.store(scales_ptr + row * (dim // 16) + groups, scale_bits, mask=valid)


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


def encode_packed_residuals(
    hidden: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    *,
    start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode local NVFP4 residuals and FP8 scales in one Triton launch."""
    if triton is None or not hidden.is_cuda:
        raise RuntimeError("fused NVFP4 encoding requires Triton and CUDA")
    batch, width, dim = hidden.shape
    if dim % 16:
        raise ValueError("hidden width must be divisible by 16")
    packed = torch.empty((batch, width, dim // 2), dtype=torch.uint8, device=hidden.device)
    scales = torch.empty((batch, width, dim // 16), dtype=torch.uint8, device=hidden.device)
    groups_per_program = 8
    _encode_nvfp4_kernel[(batch * width, triton.cdiv(dim // 16, groups_per_program))](
        hidden.contiguous(), centroids.contiguous(), labels.contiguous(),
        packed, scales, width, dim, labels.size(1), centroids.size(1), start,
        groups_per_program,
    )
    return packed, scales


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
