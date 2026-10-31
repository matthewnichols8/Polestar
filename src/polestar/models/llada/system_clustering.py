"""Quantize hidden-state residuals around fixed cluster centroids."""

from __future__ import annotations

import torch


# Positive E2M1 values, with the maximum represented by exponent 3 and mantissa 1.
_NVFP4_POSITIVE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def gather_centroid_tokens(centroids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the centroid assigned to each token in a local cache window."""
    if labels.shape[0] != centroids.shape[0] or labels.min() < 0 or labels.max() >= centroids.shape[1]:
        raise ValueError("all local token labels must identify a centroid")
    return centroids.gather(1, labels.long().unsqueeze(-1).expand(-1, -1, centroids.shape[-1]))


def quantize_residual_codes(
    residual: torch.Tensor, *, precision: str = "fp4", group_size: int = 16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize residual channels in groups of 16 using NVFP4 or symmetric INT2."""
    if residual.shape[-1] % group_size:
        raise ValueError("hidden dimension must be divisible by group_size")
    blocks = residual.float().reshape(*residual.shape[:-1], -1, group_size)
    if precision == "fp4":
        # E4M3 has three mantissa bits, so its smallest positive subnormal is 2^-9.
        scales = (blocks.abs().amax(-1) / 6.0).clamp_min(2.0 ** -9)
        scales = scales.clamp_max(torch.finfo(torch.float8_e4m3fn).max).to(torch.float8_e4m3fn)
        normalized = blocks.abs() / scales.float().unsqueeze(-1)
        levels = torch.tensor(_NVFP4_POSITIVE, device=blocks.device)
        magnitudes = (normalized.unsqueeze(-1) - levels).abs().argmin(-1).to(torch.uint8)
        codes = magnitudes + ((blocks < 0).to(torch.uint8) << 3)
    elif precision == "int2":
        scales = blocks.abs().amax(-1).clamp_min(1e-8).to(torch.float16)
        normalized = blocks / scales.float().unsqueeze(-1)
        codes = torch.bucketize(normalized.contiguous(), torch.tensor([-2 / 3, 0.0, 2 / 3], device=blocks.device)).to(torch.uint8)
    else:
        raise ValueError(f"Unsupported quantization precision: {precision}")
    return codes.reshape(*residual.shape), scales


def dequantize_residual_codes(
    codes: torch.Tensor, scales: torch.Tensor, *, precision: str = "fp4", group_size: int = 16
) -> torch.Tensor:
    """Decode E2M1 or INT2 codes and their per-group scales."""
    groups = scales.shape[-1]
    if codes.shape[-1] != groups * group_size:
        raise ValueError("code and scale shapes do not agree")
    codes = codes.reshape(*codes.shape[:-1], groups, group_size)
    if precision == "fp4":
        levels = torch.tensor(_NVFP4_POSITIVE, device=codes.device)
        values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    elif precision == "int2":
        values = (codes.float() - 1.5) / 1.5
    else:
        raise ValueError(f"Unsupported quantization precision: {precision}")
    return (values * scales.float().unsqueeze(-1)).reshape(*codes.shape[:-2], groups * group_size)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Store two 4-bit codes per byte."""
    if codes.shape[-1] % 2:
        raise ValueError("hidden dimension must be even")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    codes = torch.stack((packed & 15, packed >> 4), dim=-1)
    return codes.reshape(*packed.shape[:-1], packed.shape[-1] * 2)


@torch.no_grad()
def quantize_residual_given_clusters(
    hidden_window: torch.Tensor,
    centroids: torch.Tensor,
    labels: torch.Tensor,
    precision: str = "fp4",
    group_size: int = 16,
) -> torch.Tensor:
    """Simulate clustered residual quantization while retaining a GPU hidden cache."""
    reference = gather_centroid_tokens(centroids.float(), labels)
    codes, scales = quantize_residual_codes(
        hidden_window.float() - reference, precision=precision, group_size=group_size
    )
    residual = dequantize_residual_codes(codes, scales, precision=precision, group_size=group_size)
    return (reference + residual).to(hidden_window.dtype)
