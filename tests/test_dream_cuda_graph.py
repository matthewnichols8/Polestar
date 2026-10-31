"""Dream decode graphs must preserve projection, MLP, and cache results."""

import pytest
import torch

from polestar.models.dream.configuration_dream import DreamConfig
from polestar.models.dream.modeling_dream import DreamDecoderLayer, DreamModel
from polestar.triton_rope import triton


@pytest.mark.skipif(not torch.cuda.is_available() or triton is None, reason="CUDA and Triton required")
@pytest.mark.parametrize("fused_rope", [False, True])
def test_dream_decode_graph_matches_uncaptured_layer(fused_rope):
    torch.manual_seed(13)
    config = DreamConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    reference = DreamDecoderLayer(config, 0).eval().to("cuda", dtype=torch.bfloat16)
    captured = DreamDecoderLayer(config, 0).eval().to("cuda", dtype=torch.bfloat16)
    captured.load_state_dict(reference.state_dict())
    captured.prepare_decode_graph(8)
    captured.self_attn._fused_rope = fused_rope

    active = torch.zeros(1, 16, dtype=torch.bool, device="cuda")
    active[:, 8:] = True
    positions = torch.arange(16, device="cuda").unsqueeze(0)
    rotary_input = torch.zeros(1, 16, 16, dtype=torch.bfloat16, device="cuda")
    embeddings = reference.self_attn.rotary_emb(rotary_input, positions)

    for _ in range(2):
        hidden = torch.randn(1, 8, 64, dtype=torch.bfloat16, device="cuda")
        key = torch.randn(1, 16, 64, dtype=torch.bfloat16, device="cuda")
        value = torch.randn_like(key)
        kwargs = dict(
            position_embeddings=embeddings,
            replace_position=active,
            dual_cache=True,
            use_cache=True,
            use_cluster=False,
            layer_id=0,
        )
        expected = reference(hidden, past_key_value=(key.clone(), value.clone()), **kwargs)
        actual = captured(hidden, past_key_value=(key.clone(), value.clone()), **kwargs)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for observed, baseline in zip(actual[1], expected[1]):
            torch.testing.assert_close(observed, baseline, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available() or triton is None, reason="CUDA and Triton required")
def test_dream_generation_matches_with_graphs_and_fused_rotary():
    config = DreamConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        mask_token_id=5,
        pad_token_id=0,
        eos_token_id=2,
    )
    reference = DreamModel(config).eval().to("cuda", dtype=torch.bfloat16)
    captured = DreamModel(config).eval().to("cuda", dtype=torch.bfloat16)
    captured.load_state_dict(reference.state_dict())
    captured.prepare_decode_graphs(4)
    captured.enable_fused_rope()

    inputs = torch.tensor([[1, 7, 8]], device="cuda")
    settings = dict(
        max_new_tokens=8,
        steps=8,
        block_length=4,
        temperature=0.0,
        threshold=0.9,
        threshold_early=0.7,
        alg="confidence_threshold",
        dual_cache=True,
        use_cluster=True,
        return_dict_in_generate=True,
        output_history=False,
    )
    torch.manual_seed(23)
    expected = reference.diffusion_generate(inputs, **settings)
    torch.manual_seed(23)
    actual = captured.diffusion_generate(inputs, **settings)
    torch.testing.assert_close(actual.sequences, expected.sequences, rtol=0, atol=0)
    assert actual.nfe == expected.nfe
