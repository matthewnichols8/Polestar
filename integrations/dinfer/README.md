# Polestar in dInfer

This optional package runs Polestar-Cache and Polestar-Commit in a dInfer LLaDA
worker. The accelerated dInfer path uses the same LLaDA checkpoint as
standalone Polestar. Within dInfer's prefix decoder, each layer measures
cluster drift and passes a sparse update packet to the next layer. Final-layer
attention drift guides current-block and suffix-block token commitment.
`parallel_decoding="polestar", cache="polestar"` selects both Cache and Commit.
dInfer's threshold decoder with a prefix cache is available as a comparison
baseline; it is not a Polestar path.

The bundled dInfer code retains its [Apache-2.0 license](LICENSE).
[Upstream dInfer](https://github.com/inclusionAI/dInfer) documents its other
models and serving options.

## Install

From the release root, with a CUDA-capable PyTorch build:

```bash
python -m pip install -e '.[eval]'
python -m pip install -e integrations/dinfer
```

## Serve Polestar

```python
from dinfer.decoding.serving import DiffusionLLMServing, SamplingParams
from transformers import AutoTokenizer

model_id = "GSAI-ML/LLaDA-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "What is 12 times 8?"}],
    add_generation_prompt=True,
    tokenize=False,
)
prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
params = SamplingParams(
    parallel_decoding="polestar",
    cache="polestar",
    use_bd=False,
    mask_id=126336,
    eos_id=126081,
    polestar_seed=42,
    polestar_refresh_after_tokens=24,
    polestar_refresh_layer_count=19,
)
server = DiffusionLLMServing(
    model_id,
    model_type="llada",
    sample_params=params,
    num_gpus=1,
    dp_size=1,
    tpep_size=1,
    backend="torch",
)
try:
    output_ids = server.generate(prompt_ids, gen_length=256, block_length=32)
    print(tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True))
finally:
    server.stop_serving()
```

The adapter uses one GPU per worker and compiles the dInfer model on first use.
The first request includes compilation; a new prompt length may compile an
additional model graph. The first Cache refresh for a given sequence and window
shape captures its layer-wise update; later refreshes with that shape reuse the
graph. The worker retains up to four Cache graph shapes across requests. The
model-output KV buffer and packed-KV workspace are each shared across those
shapes and grow when a longer sequence requires more space.

`polestar_seed` defaults to 42 and seeds cache clustering for each request.
The dInfer path requires greedy decoding (`temperature=0`). By default,
Cache passes sparse updates through every model layer after more than three
committed tokens. The settings above refresh 19 of LLaDA's 31 interlayer
cache links after more than 24 committed tokens. Those links are scored
from the same cache state and patched in parallel. Both settings apply
Polestar-Cache and Polestar-Commit; the smaller, parallel refresh changes
the paper's layer-by-layer update order.

The hidden states for the local window are quantized in pinned CPU memory
and staged on GPU during each refresh. Prefix KV patches take effect on
the next model forward. The final layer's four highest-drift clusters select
suffix positions; selected positions are committed when their confidence
from the already-computed dInfer logits reaches 0.9. KV patches in dInfer
remain prefix-only. The standalone Polestar interface also caches a suffix
window; see [performance benchmarking](../../README.md#performance-benchmarking).

## Check GSM8K

Run two questions first:

```bash
python integrations/dinfer/benchmarks/evaluate_gsm8k.py \
  --backend polestar --config configs/llada8b.json --task gsm8k \
  --limit 2 --seed 42 --polestar-refresh-after-tokens 24 \
  --polestar-refresh-layers 19 \
  --output-path results/dinfer-polestar-smoke
```

For a larger comparison, run `--limit 16` with a new output directory. The
benchmark uses the same 5-shot prompts, generation length 256, and scoring
as `evaluation/run.py`. It writes accuracy to `results.json`, responses to
`samples_gsm8k.jsonl`, and tokens per second and request times to
`serving_metrics.json`. To measure dInfer alone on the same questions:

```bash
python integrations/dinfer/benchmarks/evaluate_gsm8k.py \
  --backend dinfer --config configs/llada8b.json --task gsm8k \
  --limit 2 --seed 42 --output-path results/dinfer-baseline-smoke
```

Both decoders compile their model on the first request. Add
`--prewarm-request` to both commands to run each prompt once before timing
it. The timing file also records whether the repeated answers match.
