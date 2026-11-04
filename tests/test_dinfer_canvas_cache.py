"""Polestar-Cache across DiffusionGemma denoising steps (CPU)."""

import pytest
import torch

from diffusion_gemma_tiny import CANVAS_LENGTH, TEXT_CONFIG, load_tiny_model
from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

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
    return torch.randint(3, TEXT_CONFIG["vocab_size"], (1, CANVAS_LENGTH), generator=g)


def _changed(canvas, positions):
    out = canvas.clone()
    out[0, positions] = (out[0, positions] + 1) % TEXT_CONFIG["vocab_size"]
    return out


def test_full_step_equals_the_normal_pass(model, prefix):
    canvas = _canvas(0)
    cache = CanvasPolestarCache(L)
    assert torch.equal(model.denoise(canvas, prefix, layer_runner=cache), model.denoise(canvas, prefix))
    assert cache.stats() == {"full_steps": 1, "partial_steps": 0, "recompute_frac": []}


class _Recorder:
    """Layer runner that records every layer's input of a normal pass."""

    def __init__(self):
        self.inputs = []

    def begin_step(self, canvas):
        pass

    def run_layer(self, layer, h, cos, sin, cache, positions):
        self.inputs.append(h.clone())
        return layer(h, cos, sin, cache, "decode", positions)

    def end_step(self):
        pass


def test_partial_step_with_every_cluster_selected_matches_full_recompute(model, prefix):
    # Per layer on identical inputs: the random tiny model amplifies tiny differences through depth.
    first, second = _canvas(0), _changed(_canvas(0), [1, 5, 9])
    cache = CanvasPolestarCache(L, clusters=4, topk=4, refresh_every=100)
    model.denoise(first, prefix, layer_runner=cache)
    recorder = _Recorder()
    model.denoise(second, prefix, layer_runner=recorder)
    cache.begin_step(second)
    assert not cache.full
    start = prefix.seq_length()
    positions = torch.arange(start, start + CANVAS_LENGTH)
    rope = model._rope(positions)
    for layer, h in zip(model.layers, recorder.inputs):
        cos, sin = rope[layer.layer_type]
        partial = cache.run_layer(layer, h, cos, sin, prefix, positions)
        torch.testing.assert_close(partial, layer(h, cos, sin, prefix, "decode", positions), atol=1e-5, rtol=1e-5)
    cache.end_step()
    assert cache.step_fracs == [1.0]


def test_partial_step_reuses_unselected_tokens(model, prefix):
    first, second = _canvas(0), _changed(_canvas(0), [3])
    cache = CanvasPolestarCache(L, clusters=4, topk=1, refresh_every=100)
    model.denoise(first, prefix, layer_runner=cache)
    logits = model.denoise(second, prefix, layer_runner=cache)
    assert torch.isfinite(logits).all()
    assert 0 < cache.step_fracs[0] < 1


def test_selection_is_dirty_tokens_plus_the_top_clusters(model, prefix):
    first, second = _canvas(0), _changed(_canvas(0), [2, 11])
    cache = CanvasPolestarCache(L, clusters=4, topk=1, refresh_every=100)
    model.denoise(first, prefix, layer_runner=cache)
    cache.begin_step(second)
    layer = model.layers[0]
    start = prefix.seq_length()
    cos, sin = model._rope(torch.arange(start, start + CANVAS_LENGTH))[layer.layer_type]
    keys, _ = layer.self_attn.keys_values(layer.input_norm(cache.h_in[0][None]), cos, sin, prefix)
    idx = cache._select(0, layer, keys)
    selected = set(idx.tolist())
    assert {2, 11} <= selected
    extra = selected - {2, 11}
    assert len({int(cache.labels[0][j]) for j in extra}) <= 1      # topk=1: one cluster besides dirty tokens
    assert idx.tolist() == sorted(selected)


def test_refresh_schedule():
    cache = CanvasPolestarCache(L, refresh_every=3)
    canvas = _canvas(0)
    flags = []
    for _ in range(7):
        cache.begin_step(canvas)
        flags.append(cache.full)
        cache.end_step()
    assert flags == [True, False, False, True, False, False, True]
    only_first = CanvasPolestarCache(L, refresh_every=0)
    flags = []
    for _ in range(4):
        only_first.begin_step(canvas)
        flags.append(only_first.full)
    assert flags == [True, False, False, False]


def test_reuse_modes():
    h_old, out_old, h_now = torch.ones(4, 3), torch.full((4, 3), 5.0), torch.full((4, 3), 2.0)
    for reuse, expected in [("state", out_old), ("delta", h_now * 0.5 + (out_old - h_old * 0.5))]:
        cache = CanvasPolestarCache(1, reuse=reuse)
        cache.h_in[0], cache.h_out[0] = h_old, out_old
        torch.testing.assert_close(cache._reused(0, h_now, 0.5), expected)


def test_sparse_dequant_matches_the_full_expert_pass(model):
    experts = model.layers[1].experts
    g = torch.Generator().manual_seed(5)
    y = torch.randn(3, TEXT_CONFIG["hidden_size"], generator=g)
    weights = torch.rand(3, TEXT_CONFIG["top_k_experts"], generator=g)
    ids = torch.tensor([[0, 3], [3, 6], [6, 0]])
    full = experts(y, weights, ids)
    experts.sparse_dequant = True
    try:
        sparse = experts(y, weights, ids)
    finally:
        experts.sparse_dequant = False
    torch.testing.assert_close(sparse, full)


def test_same_seed_gives_the_same_clusters(model, prefix):
    canvas = _canvas(4)
    labels = []
    for _ in range(2):
        cache = CanvasPolestarCache(L, clusters=4, polestar_seed=11)
        model.denoise(canvas, prefix, layer_runner=cache)
        labels.append([cache.labels[i].clone() for i in range(L)])
    assert all(torch.equal(a, b) for a, b in zip(*labels))


def test_canvas_llm_with_the_cache(model):
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory

    decoder = EntropyBoundParallelDecoder(TEXT_CONFIG["vocab_size"], max_denoising_steps=5, seed=0, device="cpu")
    cache = CanvasPolestarCache(L, clusters=4, topk=2, refresh_every=3)
    llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                             canvas_cache=cache)
    prompt = torch.tensor([[2, 9, 14, 21]])
    out = llm.generate(prompt, gen_length=CANVAS_LENGTH)
    assert torch.equal(out[:, :4], prompt)
    per_canvas = llm.last_stats["canvas_cache"]
    assert len(per_canvas) == len(llm.last_stats["steps_per_canvas"])
    assert per_canvas[0]["full_steps"] + per_canvas[0]["partial_steps"] == llm.last_stats["steps_per_canvas"][0]


@pytest.mark.parametrize("kwargs", [{"reuse": "both"}, {"clusters": 0}, {"refresh_every": -1}, {"topk": [1, 2]}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        CanvasPolestarCache(L, **kwargs)
