"""The dInfer path must apply selective Cache and attention-KL Commit."""

import torch
import pytest

from dinfer.decoding.utils import KVCache, TokenArray
from dinfer.model.configuration_llada import LLaDAConfig, ModelConfig
from dinfer.model.modeling_llada import LLaDAModel, LLaDAModelLM
from polestar.integrations import dinfer_commit
from polestar.integrations.dinfer_cache import PolestarCache


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


def _generator(model, *, refresh_after_tokens=3, request_seed=42):
    return dinfer_commit.make_polestar_drift_generator(
        model, threshold=0.9, lower_bound=0.7, mask_id=31, eos_id=30,
        refresh_after_tokens=refresh_after_tokens, request_seed=request_seed,
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


def test_suffix_commit_requires_drift_selection_and_high_confidence():
    logits = torch.tensor(
        [[[6.0, 0.0, 0.0], [6.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
    )
    tokens = torch.tensor([[31, 31, 31]])
    eligible = torch.tensor([True, False, True])
    selected, transfer = dinfer_commit._select_suffix_commits.__wrapped__(
        logits, tokens, eligible, mask_id=31, threshold=0.9
    )
    assert transfer.tolist() == [[True, False, False]]
    assert selected.tolist() == [[0, 31, 31]]


def test_first_block_can_select_suffix_without_prefix_cache():
    torch.manual_seed(21)
    model = _tiny_model()
    tokens = torch.randint(0, 30, (1, 12))
    cache = PolestarCache(model)
    with torch.inference_mode():
        prefill = model(tokens, use_cache=True, output_polestar_states="all")
        cache.begin_block(prefill, (0, 4))
        packed = KVCache(prefill.past_key_values)
        packed.consolidate()
        changed = tokens.clone()
        changed[:, :4] = (changed[:, :4] + 1) % 30
        dense = model(
            changed, past_key_values=packed, replace_position=(0, 12),
            use_cache=True,
        )
        cache.record_commits(4)
        cache.after_dense_step(dense, packed)
    assert cache.refresh_count == 0
    assert cache.suffix_eligible is not None
    assert cache.suffix_eligible.shape == (4,)


def test_initial_proxy_scores_do_not_require_rotary_keys(monkeypatch):
    torch.manual_seed(22)
    model = _tiny_model()
    tokens = torch.randint(0, 30, (1, 12))
    cache = PolestarCache(model)
    with torch.inference_mode():
        prefill = model(tokens, use_cache=True, output_polestar_states="all")
        cache.begin_block(prefill, (4, 8))
        raw_kv = prefill.past_key_values[:2]
        _, expected_scores = cache._proxy(0, raw_kv, use_previous=False)

        def unexpected_rotary(*_args, **_kwargs):
            raise AssertionError("initial proxy scores must not rotate cached keys")

        monkeypatch.setattr(cache.blocks[0].rotary_emb, "forward", unexpected_rotary)
        _, scores = cache._proxy(0, raw_kv, use_previous=False)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)


def test_decoder_writes_only_selected_suffix_positions(monkeypatch):
    monkeypatch.setattr(
        dinfer_commit, "_select_commits", dinfer_commit._select_commits.__wrapped__
    )
    monkeypatch.setattr(
        dinfer_commit, "_select_suffix_commits",
        dinfer_commit._select_suffix_commits.__wrapped__,
    )
    decoder = dinfer_commit.PolestarDriftDecoder(
        threshold=0.9, lower_bound=0.7, mask_id=31, eos_id=30
    )
    decoder.first_block_start = 1
    decoder.nominal_block_length = 4
    tokens = TokenArray(
        torch.tensor([[1]]), 8, mask_id=31, eos_id=30, device=torch.device("cpu")
    )
    decoder.block_init(tokens[:, 1:5], block_id=0)
    decoder.suffix_eligible = torch.tensor([True, False, True, False])
    decoder.suffix_logits = torch.zeros(1, 4, 32)
    decoder.suffix_logits[..., 2] = 6
    current_logits = torch.zeros(1, 4, 32)
    current_logits[..., 3] = 6
    decoder.decode(current_logits, 1, 5, tokens)
    assert tokens[:, 5:9].tolist() == [[2, 31, 2, 31]]


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


def test_selective_cache_patches_every_layer_and_changes_next_logits(monkeypatch):
    torch.manual_seed(4)
    model = _tiny_model(n_layers=4)
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
        assert len(prefill.hidden_states) == 4
        assert len(bridge.cache.states) == 4
        assert all(
            state.offload.q_cpu.device.type == "cpu"
            for state in bridge.cache.states
        )
        assert generator.decoder.current_attention.shape == (1, 2, 4, 16)
        packed = KVCache(prefill.past_key_values)
        packed.consolidate()
        unpatched = KVCache(prefill.past_key_values)
        unpatched.consolidate()
        old_keys = [packed.get_keys(layer).clone() for layer in range(1, 4)]
        changed = tokens[:, 4:].clone()
        changed[:, :4] = (changed[:, :4] + 1) % 30
        dense = model(
            changed, past_key_values=packed, replace_position=(4, 16), use_cache=True
        )
        for layer in range(4):
            torch.testing.assert_close(
                dense.past_key_values[2 * layer][:, :, :4],
                packed.get_keys(layer)[:, :, :4],
                rtol=0,
                atol=0,
            )
        bridge.cache.record_commits(3)
        bridge(
            changed, past_key_values=packed, replace_position=(4, 16), use_cache=True
        )
        assert bridge.cache.refresh_count == 0
        bridge.cache.record_commits(1)
        refreshed = bridge(
            changed, past_key_values=packed, replace_position=(4, 16), use_cache=True
        )
        for layer in range(1, 4):
            torch.testing.assert_close(
                refreshed.past_key_values[2 * layer][:, :, :4],
                packed.get_keys(layer)[:, :, :4],
                rtol=0,
                atol=0,
            )
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
    assert [
        torch.equal(old, packed.get_keys(layer))
        for layer, old in enumerate(old_keys, start=1)
    ] == [False, False, False]
    assert not torch.equal(logits_with_cache, logits_without_cache)
    with torch.inference_mode():
        bridge.cache.record_commits(4)
        bridge(next_tokens, past_key_values=packed, replace_position=(4, 16), use_cache=True)
    assert bridge.cache.refresh_count == 2


def test_compact_outputs_match_full_model_for_prefill_and_suffix():
    torch.manual_seed(13)
    model = _tiny_model(n_layers=3)
    tokens = torch.randint(0, 30, (1, 16))
    with torch.inference_mode():
        full = model(tokens, use_cache=True, output_hidden_states=True)
        compact = model(tokens, use_cache=True, output_polestar_states=True)
        final_only = model(tokens, use_cache=True, output_polestar_states="final")
    torch.testing.assert_close(full.logits, compact.logits, rtol=0, atol=0)
    assert len(compact.hidden_states) == 3
    assert len(final_only.hidden_states) == 1
    torch.testing.assert_close(final_only.hidden_states[0], full.hidden_states[-2], rtol=0, atol=0)
    for observed, expected in zip(compact.hidden_states, full.hidden_states[:-1]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
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
        final_decode = model(
            suffix, past_key_values=compact_cache, replace_position=(4, 16),
            use_cache=True, output_polestar_states="final",
        )
    torch.testing.assert_close(full_decode.logits, compact_decode.logits, rtol=0, atol=0)
    for observed, expected in zip(compact_decode.hidden_states, full_decode.hidden_states[:-1]):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        final_decode.hidden_states[0], full_decode.hidden_states[-2], rtol=0, atol=0
    )


def test_final_block_keeps_all_eight_cache_clusters():
    torch.manual_seed(19)
    model = _tiny_model()
    tokens = torch.randint(0, 30, (1, 16))
    cache = PolestarCache(model)
    with torch.inference_mode():
        output = model(tokens, use_cache=True, output_polestar_states=True)
        cache.begin_block(output, (12, 16))
    assert all(state.clusters[0].size(1) == 8 for state in cache.states)
    with torch.inference_mode():
        short_output = model(tokens[:, :5], use_cache=True, output_polestar_states=True)
        cache.begin_block(short_output, (4, 5))
    assert all(state.clusters[0].size(1) == 2 for state in cache.states)


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_captured_refresh_matches_eager_across_two_updates():
    torch.manual_seed(41)
    model = _tiny_model(n_layers=4).cuda()
    tokens = torch.randint(0, 30, (1, 16), device="cuda")
    changed = tokens[:, 4:].clone()
    changed[:, :4] = (changed[:, :4] + 1) % 30
    with torch.inference_mode():
        prefill = model(tokens, use_cache=True, output_polestar_states=True)

        def run(captured):
            cache = PolestarCache(model, use_cuda_graph=captured)
            packed = KVCache(prefill.past_key_values)
            packed.consolidate()
            torch.manual_seed(43)
            cache.begin_block(prefill, (4, 8))
            for _ in range(2):
                output = model(
                    changed, past_key_values=packed, replace_position=(4, 16),
                    use_cache=True,
                )
                cache.record_commits(4)
                cache.after_dense_step(output, packed)
            if cache.flat_offload.ready_event is not None:
                cache.flat_offload.ready_event.synchronize()
            return (
                packed._data.clone(),
                torch.stack([layer.clusters[0] for layer in cache.states]),
                cache.flat_offload.q_cpu.clone(),
                cache.flat_offload.scales_cpu.clone(),
                cache.effective_patched_rows.clone(),
            )

        eager = run(False)
        captured = run(True)
    for actual, expected in zip(captured, eager):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_captured_refresh_reuses_graph_across_blocks_and_requests():
    torch.manual_seed(61)
    model = _tiny_model(n_layers=4).cuda()
    tokens_by_length = {
        length: torch.randint(0, 30, (1, length), device="cuda")
        for length in (36, 32)
    }

    def run(captured):
        cache = PolestarCache(model, use_cuda_graph=captured)
        snapshots = []
        graph_by_length = {}
        shared_addresses = []
        for length in (36, 32, 36):
            if isinstance(cache.effective_patched_rows, torch.Tensor):
                with torch.inference_mode():
                    cache.effective_patched_rows.zero_()
            for start in (12, 16):
                with torch.inference_mode():
                    tokens = tokens_by_length[length]
                    prefill = model(
                        tokens, use_cache=True, output_polestar_states=True
                    )
                    packed = KVCache(prefill.past_key_values)
                    packed.consolidate()
                    torch.manual_seed(71 + length + start)
                    cache.begin_block(prefill, (start, start + 4))
                    changed = tokens[:, start:].clone()
                    changed[:, :4] = (changed[:, :4] + 1) % 30
                    for _ in range(2):
                        output = model(
                            changed,
                            past_key_values=packed,
                            replace_position=(start, length),
                            use_cache=True,
                        )
                        cache.record_commits(4)
                        cache.after_dense_step(output, packed)
                    if cache.flat_offload.ready_event is not None:
                        cache.flat_offload.ready_event.synchronize()
                    snapshots.append((
                        packed._data.clone(),
                        torch.stack([layer.clusters[0] for layer in cache.states]),
                        cache.flat_offload.q_cpu.clone(),
                        cache.flat_offload.scales_cpu.clone(),
                        cache.effective_patched_rows.clone(),
                    ))
                    if captured:
                        workspace = cache._active_workspace
                        if length in graph_by_length:
                            assert workspace.graph is graph_by_length[length]
                        else:
                            graph_by_length[length] = workspace.graph
                        shared_addresses.append((
                            workspace.key_values[0].untyped_storage().data_ptr(),
                            workspace.packed._data.untyped_storage().data_ptr(),
                        ))
            cache.finish_request()
        if captured:
            assert len(cache._graph_pool) == 2
            assert len(set(shared_addresses)) == 1
        return snapshots

    eager = run(False)
    captured = run(True)
    for actual_block, expected_block in zip(captured, eager):
        for actual, expected in zip(actual_block, expected_block):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
