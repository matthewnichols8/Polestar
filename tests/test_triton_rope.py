import pytest
import torch

from polestar.models.llada.triton_rope import apply_rotary, triton


@pytest.mark.skipif(not torch.cuda.is_available() or triton is None, reason="CUDA and Triton required")
@pytest.mark.parametrize("length", [32, 256])
def test_fused_rope_matches_full_precision_reference(length):
    torch.manual_seed(19)
    x = torch.randn(1, length, 32, 128, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    angles = torch.randn(length + 37, 128, device="cuda", dtype=torch.float32)
    sin, cos = angles.sin()[None, None], angles.cos()[None, None]
    rotated = torch.cat((-x.float()[..., 64:], x.float()[..., :64]), dim=-1)

    start = 37
    expected = (x.float() * cos[..., start:start + length, :]
                + rotated * sin[..., start:start + length, :]).to(x.dtype)
    actual = apply_rotary(x, sin, cos, position_start=start)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    positions = torch.randperm(length + 37, device="cuda")[:length]
    selected = (x.float() * cos[:, :, positions, :]
                + rotated * sin[:, :, positions, :]).to(x.dtype)
    indexed = apply_rotary(x, sin, cos, positions=positions)
    torch.testing.assert_close(indexed, selected, rtol=0, atol=0)
