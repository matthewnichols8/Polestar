"""Polestar-Commit (drift-gated acceptance) for DiffusionGemma's canvas in dInfer (CPU)."""

import numpy as np
import pytest
import torch

from diffusion_gemma_tiny import CANVAS_LENGTH, TEXT_CONFIG, load_tiny_model
from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache
from polestar.integrations.dinfer_canvas_commit import (
    CanvasDriftTracker,
    CommitTraceRecorder,
    PolestarCommitEntropyDecoder,
    select_canvas_drift_commits,
)

V = TEXT_CONFIG["vocab_size"]
L = TEXT_CONFIG["num_hidden_layers"]


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return load_tiny_model(tmp_path_factory.mktemp("dgemma"))


@pytest.fixture()
def prefix(model):
    cache = model.new_cache()
    model.encode(torch.tensor([[2, 9, 14, 21, 33, 40]]), cache)
    return cache


def _canvas(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, V, (1, CANVAS_LENGTH), generator=g)


def _base_decoder(seed=0, **kwargs):
    from dinfer import EntropyBoundParallelDecoder

    return EntropyBoundParallelDecoder(V, seed=seed, device="cpu", **kwargs)


def _commit_decoder(seed=0, **kwargs):
    return PolestarCommitEntropyDecoder(V, seed=seed, device="cpu", **kwargs)


def _probs(seed, S=24):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(1, 4, CANVAS_LENGTH, S, generator=g) * 3, dim=-1)


# ----------------------------------------------------------------------------- attention probe

def test_attention_probs_are_the_distribution_attend_uses(model, prefix):
    layer = model.layers[-1]
    attn = layer.self_attn
    start = prefix.seq_length()
    cos, sin = model._rope(torch.arange(start, start + CANVAS_LENGTH))[layer.layer_type]
    h = torch.randn(1, CANVAS_LENGTH, TEXT_CONFIG["hidden_size"], generator=torch.Generator().manual_seed(1))
    x = layer.input_norm(h)
    keys, values = attn.keys_values(x, cos, sin, prefix)
    probs = attn.attention_probs(x, cos, sin, keys)
    assert probs.shape == (1, attn.num_heads, CANVAS_LENGTH, keys.shape[2])
    torch.testing.assert_close(probs.sum(-1), torch.ones_like(probs[..., 0]))
    v = values.float().repeat_interleave(attn.num_heads // attn.num_kv_heads, dim=1)
    out = torch.matmul(probs, v).transpose(1, 2).reshape(1, CANVAS_LENGTH, -1)
    from dinfer.model.modeling_diffusion_gemma import linear16

    torch.testing.assert_close(linear16(out, attn.o_proj.weight), attn.attend(x, cos, sin, keys, values),
                               atol=1e-4, rtol=1e-4)


def test_the_probe_does_not_change_the_logits(model, prefix):
    seen = []
    for seed in range(2):
        canvas = _canvas(seed)
        assert torch.equal(model.denoise(canvas, prefix), model.denoise(canvas, prefix, attention_probe=seen.append))
    assert len(seen) == 2 and seen[0].shape[2] == CANVAS_LENGTH


def test_the_probe_does_not_change_polestar_cache_steps(model, prefix):
    a, b = CanvasPolestarCache(L, refresh_every=0, topk=1), CanvasPolestarCache(L, refresh_every=0, topk=1)
    seen = []
    for seed in range(3):   # step 0 is a full step, then partial steps
        canvas = _canvas(seed)
        assert torch.equal(model.denoise(canvas, prefix, layer_runner=a),
                           model.denoise(canvas, prefix, layer_runner=b, attention_probe=seen.append))
    assert b.stats()["partial_steps"] == 2 and len(seen) == 3


# ----------------------------------------------------------------------------- drift and the commit rule

def test_drift_tracker_window_and_reset():
    t = CanvasDriftTracker(window=2)
    p0, p1, p2, p3 = (_probs(s) for s in range(4))
    assert t.update(p0) == (None, None)                     # nothing to compare with yet
    d1, delta1 = t.update(p1)
    assert torch.equal(delta1, d1)                          # first drift: delta = drift
    d2, delta2 = t.update(p2)
    torch.testing.assert_close(delta2, d2 - d1)
    d3, delta3 = t.update(p3)
    torch.testing.assert_close(delta3, d3 - (d1 + d2) / 2)  # mean of the last `window` drifts
    t.reset()
    assert t.update(p0) == (None, None)
    assert t.update(_probs(5, S=30)) == (None, None)       # a shape change restarts the canvas


def test_commit_rule_band_gate_and_cap():
    conf = torch.tensor([0.95, 0.86, 0.86, 0.70, 0.75, 0.75, 0.60])
    delta = torch.tensor([9.0, 0.02, 0.01, 9.0, 0.30, 2.50, 9.0])
    got = select_canvas_drift_commits(conf, delta, threshold=0.9, lower_bound=0.7, gate_scale=10.0, drift_cap=2.0)
    # 0.95: above the band (the entropy rule's job)   0.86: gate 0.016 -> 0.02 passes, 0.01 does not
    # 0.70: not strictly above the lower bound       0.75: gate 0.225 -> 0.30 passes, 2.5 is capped
    # 0.60: below the band
    assert got.tolist() == [False, True, False, False, True, False, False]


# ----------------------------------------------------------------------------- the decoder

def _run(decoder, logits_seq, attentions=None):
    x = torch.zeros(1, CANVAS_LENGTH, dtype=torch.long)
    decoder.init_block(x, 0, CANVAS_LENGTH)
    outs = []
    for i, logits in enumerate(logits_seq):
        if attentions is not None:
            decoder.observe_attention(attentions[i])
        done = decoder.decode(logits, 0, CANVAS_LENGTH, x, decoder.max_denoising_steps - i)
        outs.append((x.clone(), decoder.argmax.clone(), done.clone(), decoder.accepted.clone()))
    return outs


def _logit_sequence(n=6):
    g = torch.Generator().manual_seed(11)
    base = torch.randn(1, CANVAS_LENGTH, V, generator=g)
    target = torch.arange(CANVAS_LENGTH) % V
    sharp = torch.full((1, CANVAS_LENGTH, V), -30.0)
    sharp[0, torch.arange(CANVAS_LENGTH), target] = 30.0
    return [base * (1 + i) for i in range(n - 2)] + [sharp, sharp]   # last two steps make the stop rule fire


@pytest.mark.parametrize("mode", ["off", "record"])
def test_off_and_record_modes_match_the_entropy_bound_decoder(mode, tmp_path):
    logits_seq = _logit_sequence()
    attentions = [_probs(s) for s in range(len(logits_seq))]
    recorder = CommitTraceRecorder(str(tmp_path)) if mode == "record" else None
    ref = _run(_base_decoder(seed=7), logits_seq)
    got = _run(_commit_decoder(seed=7, commit_mode=mode, recorder=recorder), logits_seq, attentions)
    for (rx, ra, rd, racc), (gx, ga, gd, gacc) in zip(ref, got):
        assert torch.equal(rx, gx) and torch.equal(ra, ga) and torch.equal(rd, gd) and torch.equal(racc, gacc)
    assert bool(ref[-1][2].all())   # the stop rule fired, so it was compared too


def test_accept_mode_adds_drift_commits_and_never_removes_entropy_acceptances():
    logits_seq = _logit_sequence(n=4)
    attentions = [_probs(s) for s in range(4)]
    base = _run(_commit_decoder(seed=3, commit_mode="record"), logits_seq, attentions)
    # a permissive band and gate so that some positions qualify
    decoder = _commit_decoder(seed=3, commit_mode="accept", commit_lower_bound=0.0, commit_threshold=1.0,
                              commit_gate_scale=0.0, commit_drift_cap=1e9)
    got = _run(decoder, logits_seq, attentions)
    assert not got[0][3].sum() > base[0][3].sum()          # no drift on the first step
    added = 0
    for (_, _, _, b_acc), (_, _, _, g_acc) in zip(base, got):
        assert bool((g_acc | ~b_acc).all())                 # every entropy-bound acceptance is kept
        added += int((g_acc & ~b_acc).sum())
    assert added > 0
    stats = decoder.end_canvas(got[-1][1])
    assert stats["mode"] == "accept" and stats["steps"] == 4 and stats["extra_accepted"] > 0


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        _commit_decoder(commit_mode="freeze")
    with pytest.raises(ValueError):
        _commit_decoder(commit_confidence="median")
    with pytest.raises(ValueError):
        _commit_decoder(commit_lower_bound=0.9, commit_threshold=0.7)
    with pytest.raises(ValueError):
        CanvasDriftTracker(window=0)


# ----------------------------------------------------------------------------- end to end

@pytest.mark.parametrize("with_cache", [False, True])
@pytest.mark.parametrize("mode", ["record", "accept"])
def test_canvas_llm_with_commit(model, tmp_path, mode, with_cache):
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, KVCacheFactory

    recorder = CommitTraceRecorder(str(tmp_path)) if mode == "record" else None
    decoder = _commit_decoder(commit_mode=mode, max_denoising_steps=4, recorder=recorder)
    canvas_cache = CanvasPolestarCache(L, refresh_every=2) if with_cache else None
    llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                             canvas_cache=canvas_cache)
    prompt = torch.tensor([[2, 9, 14, 21, 33, 40]])
    if recorder is not None:
        recorder.begin_request("q0")
    out = llm.generate(prompt, gen_length=2 * CANVAS_LENGTH)
    stats = llm.last_stats
    assert torch.equal(out[:, :6], prompt)
    assert len(stats["commit"]) == len(stats["steps_per_canvas"])
    assert [c["steps"] for c in stats["commit"]] == stats["steps_per_canvas"]
    if mode == "record":
        trace = np.load(stats["commit"][0]["trace"])
        T = stats["steps_per_canvas"][0]
        for key in ("conf_raw", "conf_tempered", "entropy", "drift", "base_accepted", "accepted", "argmax", "sampled"):
            assert trace[key].shape == (T, CANVAS_LENGTH), key
        assert np.isnan(trace["drift"][0]).all()           # no drift on a canvas's first step
        assert np.array_equal(trace["base_accepted"], trace["accepted"])
        assert trace["final"].shape == (CANVAS_LENGTH,) and 0 <= int(trace["valid_len"]) <= CANVAS_LENGTH


# ----------------------------------------------------------------------------- full Commit: freeze + Cache refresh

def _two_targets(n_frozen=10):
    """Step 0 is very confident about token A at the first `n_frozen` positions, step 1 about token B."""
    g = torch.Generator().manual_seed(5)
    steps = []
    for target in (3, 4):
        logits = torch.randn(1, CANVAS_LENGTH, V, generator=g)
        logits[0, :n_frozen] = -30.0
        logits[0, :n_frozen, target] = 30.0
        steps.append(logits)
    return steps


def test_freeze_commits_permanently_to_the_argmax():
    counts = []
    decoder = _commit_decoder(seed=1, commit_mode="off", commit_freeze_threshold=0.999)
    decoder.commit_callback = counts.append
    out = _run(decoder, _two_targets())
    for x, argmax, _, accepted in out:
        assert bool((x[0, :10] == 3).all()) and bool((argmax[0, :10] == 3).all())   # A, never B
        assert bool(accepted[0, :10].all())
    assert counts[0] >= 10 and counts[1] == 0      # nothing new: B is just as confident but already frozen
    assert bool(decoder.frozen[0, :10].all())
    assert decoder.end_canvas(out[-1][1])["frozen"] == counts[0]


def test_without_freeze_the_same_positions_follow_the_model():
    out = _run(_commit_decoder(seed=1, commit_mode="off"), _two_targets())
    assert bool((out[1][1][0, :10] == 4).all())


def test_freeze_resets_per_canvas_and_is_recorded(tmp_path):
    decoder = _commit_decoder(seed=1, commit_mode="record", commit_freeze_threshold=0.999,
                              recorder=CommitTraceRecorder(str(tmp_path)))
    out = _run(decoder, _two_targets(), [_probs(s) for s in range(2)])
    trace = np.load(decoder.end_canvas(out[-1][1])["trace"])
    assert trace["frozen"][:, :10].all() and (trace["argmax"][1, :10] == 4).all()   # the model's argmax is traced
    decoder.init_block(torch.zeros(1, CANVAS_LENGTH, dtype=torch.long), 0, CANVAS_LENGTH)
    assert decoder.frozen is None


def test_cache_refreshes_after_commits():
    cache = CanvasPolestarCache(L, refresh_every=0, refresh_after_commits=3)
    fulls = []
    for commits in (0, 3, 4, 0, 1):          # reported after each step
        cache.begin_step(_canvas(0))
        fulls.append(cache.full)
        cache.record_commits(commits)
    # step 0 full; 0 commits -> partial; 3 (not more than 3) -> partial; 3+4 -> full (resets); 0 -> partial
    assert fulls == [True, False, False, True, False]
    assert cache.commit_refreshes == 1
    cache.reset()
    assert cache.commits_since_refresh == 0


def test_invalid_freeze_and_refresh_settings_are_rejected():
    with pytest.raises(ValueError):
        _commit_decoder(commit_freeze_threshold=1.5)
    with pytest.raises(ValueError):
        CanvasPolestarCache(L, refresh_after_commits=-1)


def test_canvas_llm_with_full_commit_drives_cache_refreshes(model):
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, KVCacheFactory

    decoder = _commit_decoder(commit_mode="accept", commit_freeze_threshold=0.5, max_denoising_steps=4)
    cache = CanvasPolestarCache(L, refresh_every=0, refresh_after_commits=0)
    llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                             canvas_cache=cache)
    assert decoder.commit_callback == cache.record_commits
    prompt = torch.tensor([[2, 9, 14, 21, 33, 40]])
    out = llm.generate(prompt, gen_length=2 * CANVAS_LENGTH)
    stats = llm.last_stats
    assert torch.equal(out[:, :6], prompt)
    frozen = sum(c["frozen"] for c in stats["commit"])
    refreshes = sum(c["commit_refreshes"] for c in stats["canvas_cache"])
    assert (frozen > 0) == (refreshes > 0)
