"""The dInfer path must apply selective Cache and attention-KL Commit."""

import torch

from dinfer.decoding.utils import KVCache, TokenArray
from dinfer.model.configuration_llada import LLaDAConfig, ModelConfig
from dinfer.model.modeling_llada import LLaDAModel, LLaDAModelLM
from polestar.integrations import dinfer_commit


def _tiny_model(n_layers=2):
    settings = dict(
        d_model=16, n_heads=2, n_layers=n_layers, embedding_size=32,
        vocab_size=32, mlp_ratio=2, rope=True, flash_attention=False,
        max_sequence_length=32, block_type="llama", activation_type="silu",
        layer_norm_type="rms", residual_dropout=0.0,
    )
    return LLaDAModelLM(
        LLaDAConfig(**settings),
        model=LLaDAModel(ModelConfig(**settings), init_params=True),
    ).eval()


def _generator(model, *, cache_interval=2, request_seed=42):
    return dinfer_commit.make_polestar_drift_generator(
        model, threshold=0.9, lower_bound=0.7, mask_id=31, eos_id=30,
        cache_interval=cache_interval, request_seed=request_seed,
    )


def test_drift_gate_adds_subthreshold_commit():
    logits = torch.tensor([[[2.0, 0.0, 0.0], [1.8, 0.0, 0.0]]])
    tokens = torch.full((1, 2), 2)
    drift = torch.full((1, 2), 0.3)
    select = dinfer_commit._select_commits.__wrapped__
    _, high_only = select(
        logits, tokens, drift, mask_id=2, threshold=0.9,
        lower_bound=0.7, allow_drift=False,
    )
    _, with_drift = select(
        logits, tokens, drift, mask_id=2, threshold=0.9,
        lower_bound=0.7, allow_drift=True,
    )
    assert high_only.tolist() == [[True, False]]
    assert with_drift.tolist() == [[True, True]]


def test_decoder_uses_attention_kl_without_copying_fresh_probabilities(monkeypatch):
    monkeypatch.setattr(
        dinfer_commit, "_select_commits", dinfer_commit._select_commits.__wrapped__
    )
    decoder = dinfer_commit.PolestarDriftDecoder(
        threshold=0.9, lower_bound=0.7, mask_id=31, eos_id=30
    )
    decoder.first_block_start = 1
    decoder.nominal_block_length = 4
    tokens = TokenArray(
        torch.tensor([[1]]), 4, mask_id=31, eos_id=30, device=torch.device("cpu")
    )
    decoder.block_init(tokens[:, 1:5], block_id=0)
    logits = torch.zeros(1, 4, 32)
    logits[..., 2] = 5
    attention = torch.rand(1, 2, 4, 5)
    attention = attention / attention.sum(dim=-1, keepdim=True)
    decoder.current_attention = attention
    decoder.decode(logits, 1, 5, tokens)
    assert decoder.previous_attention.data_ptr() == attention.data_ptr()
    assert not decoder.drift_window
    decoder.current_attention = attention.roll(1, dims=-1)
    decoder.decode(logits, 1, 5, tokens)
    assert len(decoder.drift_window) == 1


def test_selective_cache_patches_prefix_and_changes_next_logits(monkeypatch):
    torch.manual_seed(4)
    model = _tiny_model()
    monkeypatch.setattr(
        dinfer_commit, "_select_commits", dinfer_commit._select_commits.__wrapped__
    )
    generator = _generator(model)
    bridge = generator.model
    generator.decoder.first_block_start = 4
    generator.decoder.nominal_block_length = 4
    generator.decoder.block_init(torch.full((1, 4), 31), block_id=0)
    tokens = torch.randint(0, 30, (1, 16))
    with torch.inference_mode():
        prefill = bridge(tokens, use_cache=True)
        assert len(prefill.hidden_states) == 2
        assert generator.decoder.current_attention.shape == (1, 2, 4, 16)
        packed = KVCache(prefill.past_key_values)
        packed.consolidate()
        unpatched = KVCache(prefill.past_key_values)
        unpatched.consolidate()
        old_last_key = packed.get_keys(1).clone()
        changed = tokens[:, 4:].clone()
        changed[:, :4] = (changed[:, :4] + 1) % 30
        bridge(changed, past_key_values=packed, replace_position=(4, 16), use_cache=True)
        next_tokens = changed.clone()
        next_tokens[:, :4] = (next_tokens[:, :4] + 1) % 30
        logits_with_cache = model(
            next_tokens, past_key_values=packed, replace_position=(4, 16)
        ).logits
        logits_without_cache = model(
            next_tokens, past_key_values=unpatched, replace_position=(4, 16)
        ).logits
    assert bridge.cache.refresh_count == 1
    assert bridge.cache.effective_patched_rows > 0
    assert not torch.equal(old_last_key, packed.get_keys(1))
    assert not torch.equal(logits_with_cache, logits_without_cache)


def test_compact_outputs_match_full_model_for_prefill_and_suffix():
    torch.manual_seed(13)
    model = _tiny_model(n_layers=3)
    tokens = torch.randint(0, 30, (1, 16))
    with torch.inference_mode():
        full = model(tokens, use_cache=True, output_hidden_states=True)
        compact = model(tokens, use_cache=True, output_polestar_states=True)
    torch.testing.assert_close(full.logits, compact.logits, rtol=0, atol=0)
    assert len(compact.hidden_states) == 2
    torch.testing.assert_close(compact.hidden_states[0], full.hidden_states[-3], rtol=0, atol=0)
    torch.testing.assert_close(compact.hidden_states[1], full.hidden_states[-2], rtol=0, atol=0)
    suffix = tokens[:, 4:].clone()
    suffix[:, :4] = (suffix[:, :4] + 1) % 30
    full_cache = KVCache(full.past_key_values)
    full_cache.consolidate()
    compact_cache = KVCache(full.past_key_values)
    compact_cache.consolidate()
    with torch.inference_mode():
        full_decode = model(
            suffix, past_key_values=full_cache, replace_position=(4, 16),
            use_cache=True, output_hidden_states=True,
        )
        compact_decode = model(
            suffix, past_key_values=compact_cache, replace_position=(4, 16),
            use_cache=True, output_polestar_states=True,
        )
    torch.testing.assert_close(full_decode.logits, compact_decode.logits, rtol=0, atol=0)
    torch.testing.assert_close(
        full_decode.hidden_states[-3], compact_decode.hidden_states[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        full_decode.hidden_states[-2], compact_decode.hidden_states[1], rtol=0, atol=0
    )


def test_explicit_step_cap_must_match_generation_length():
    model = _tiny_model()
    generator = dinfer_commit.make_polestar_drift_generator(
        model, threshold=0.9, lower_bound=0.7, mask_id=31, eos_id=30, steps=4,
    )
    try:
        generator.generate(torch.tensor([[1, 2, 3, 4]]), gen_length=8, block_length=4)
    except ValueError as exc:
        assert "steps=gen_length" in str(exc)
    else:
        raise AssertionError("mismatched explicit step cap was accepted")
