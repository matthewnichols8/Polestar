"""DiffusionGemma in dInfer: model, per-layer cache, entropy-bound decoder and canvas loop (CPU)."""

import pytest
import torch

from diffusion_gemma_tiny import CANVAS_LENGTH, TEXT_CONFIG, load_tiny_model


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    return load_tiny_model(tmp_path_factory.mktemp("dgemma"))


def _prompt(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, TEXT_CONFIG["vocab_size"], (1, n), generator=g)


def _decoder(seed=0, **kwargs):
    from dinfer import EntropyBoundParallelDecoder

    return EntropyBoundParallelDecoder(TEXT_CONFIG["vocab_size"], seed=seed, device="cpu", **kwargs)


def test_denoise_reads_but_does_not_write_the_cache(model):
    cache = model.new_cache()
    model.encode(_prompt(5), cache)
    before = [k.clone() for k in cache.k]
    logits = model.denoise(_prompt(CANVAS_LENGTH, seed=1), cache)
    assert logits.shape == (1, CANVAS_LENGTH, TEXT_CONFIG["vocab_size"])
    assert torch.isfinite(logits).all()
    assert cache.seq_length() == 5
    assert all(torch.equal(a, b) for a, b in zip(before, cache.k))


def test_sliding_layers_keep_only_the_window(model):
    cache = model.new_cache()
    model.encode(_prompt(12), cache)
    keep = TEXT_CONFIG["sliding_window"] - 1
    for i, layer_type in enumerate(TEXT_CONFIG["layer_types"]):
        expected = keep if layer_type == "sliding_attention" else 12
        assert cache.k[i].shape[2] == expected
    assert cache.key_positions("sliding_attention").tolist() == list(range(12 - keep, 12))
    assert cache.seq_length() == 12


def test_chunked_encode_matches_one_pass(model):
    # Compare the cached K/V directly: the random tiny model amplifies 1e-6 differences in the logits.
    prompt = _prompt(12)
    one, chunked = model.new_cache(), model.new_cache()
    model.encode(prompt, one, chunk_size=12)
    model.encode(prompt, chunked, chunk_size=4)
    for i in range(TEXT_CONFIG["num_hidden_layers"]):
        torch.testing.assert_close(chunked.k[i], one.k[i], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(chunked.v[i], one.v[i], atol=1e-5, rtol=1e-5)
    for layer_type in set(TEXT_CONFIG["layer_types"]):
        assert torch.equal(chunked.key_positions(layer_type), one.key_positions(layer_type))


def test_entropy_bound_decoder_is_seeded():
    logits = torch.randn(1, CANVAS_LENGTH, TEXT_CONFIG["vocab_size"], generator=torch.Generator().manual_seed(3))
    outputs = []
    for _ in range(2):
        decoder = _decoder(seed=7)
        x = torch.zeros(1, CANVAS_LENGTH, dtype=torch.long)
        decoder.init_block(x, 0, CANVAS_LENGTH)
        decoder.decode(logits, 0, CANVAS_LENGTH, x, decoder.max_denoising_steps)
        outputs.append((x.clone(), decoder.argmax.clone()))
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][1], logits.argmax(-1))


def test_entropy_bound_decoder_stops_when_stable_and_confident():
    V = TEXT_CONFIG["vocab_size"]
    target = torch.arange(CANVAS_LENGTH) % V
    logits = torch.full((1, CANVAS_LENGTH, V), -30.0)
    logits[0, torch.arange(CANVAS_LENGTH), target] = 30.0
    decoder = _decoder()
    x = torch.zeros(1, CANVAS_LENGTH, dtype=torch.long)
    decoder.init_block(x, 0, CANVAS_LENGTH)
    first = decoder.decode(logits, 0, CANVAS_LENGTH, x, 4)
    second = decoder.decode(logits, 0, CANVAS_LENGTH, x, 3)
    assert not bool(first.all())         # the argmax history needs one more step
    assert bool(second.all())
    assert torch.equal(x[0], target)     # every confident position was accepted


def test_canvas_llm_generates_after_the_prompt(model):
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, KVCacheFactory

    prompt = _prompt(6)
    llm = CanvasDiffusionLLM(model, _decoder(max_denoising_steps=4), BlockIteratorFactory(),
                             KVCacheFactory("prefix", per_layer=True))
    out = llm.generate(prompt, gen_length=2 * CANVAS_LENGTH)
    stats = llm.last_stats
    assert torch.equal(out[:, :6], prompt)
    assert out.shape[1] <= 6 + 2 * CANVAS_LENGTH
    assert 1 <= len(stats["steps_per_canvas"]) <= 2
    assert all(1 <= s <= 4 for s in stats["steps_per_canvas"])
    assert stats["num_forwards"] == sum(stats["steps_per_canvas"])
    assert stats["new_tokens"] == out.shape[1] - 6


def test_canvas_llm_requires_the_per_layer_prefix_cache(model):
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, KVCacheFactory

    with pytest.raises(ValueError):
        CanvasDiffusionLLM(model, _decoder(), BlockIteratorFactory(), KVCacheFactory("prefix"))
    with pytest.raises(ValueError):
        CanvasDiffusionLLM(model, _decoder(), BlockIteratorFactory(start_block_align=True),
                           KVCacheFactory("prefix", per_layer=True))
