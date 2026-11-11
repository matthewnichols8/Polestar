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

## DiffusionGemma

The bundled dInfer also runs
[`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
(text only), a uniform-noise diffusion model. Each 256-token canvas starts as random tokens instead of
masks, and every position is re-predicted on every denoising step until an entropy-bound sampler converges.
The components follow dInfer's structure:

| Component | Class |
|---|---|
| Model | `dinfer.model.DiffusionGemmaModelLM`: a causal `encode` writes the prompt and each committed canvas to the prefix cache; a bidirectional `denoise` reads it |
| Decoder | `dinfer.EntropyBoundParallelDecoder` |
| KV cache | `KVCacheFactory("prefix", per_layer=True)`: sliding-window and full-attention layers keep separate K/V |
| Iteration | `dinfer.CanvasDiffusionLLM` |

The model's experts are split across the ranks of the `torch.distributed` group (expert parallel); a single
process uses one GPU. `expert_dtype="bf16"` uses the checkpoint weights and keeps the experts that do not fit
in pinned CPU memory; `expert_dtype="int8"` quantizes the experts so that the model fits on two 24 GB GPUs.
Batch size is 1. The tokenizer works with the repository's pinned Transformers 4.49.

### Polestar-Cache for DiffusionGemma

DiffusionGemma's prefix cache is written by a causal encode, so its states never drift and there is nothing
to refresh there. The cost is in the canvas: every denoising step recomputes all canvas tokens in every
layer. `CanvasPolestarCache` applies Polestar-Cache across the denoising steps of a canvas. On a partial step,
each layer recomputes only:

- the tokens whose input changed since the previous step;
- the tokens in the `topk` clusters whose centroid attention drifted most.

Other tokens reuse their cached layer update. The canvas is recomputed in full, and re-clustered, on its first step and
every `refresh_every` steps. Partial steps dequantize only the experts that the selected tokens use.

```python
import torch
from huggingface_hub import snapshot_download

from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory
from dinfer.model import DiffusionGemmaModelLM, load_diffusion_gemma_tokenizer
from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

path = snapshot_download("google/diffusiongemma-26B-A4B-it")
model = DiffusionGemmaModelLM.from_pretrained(path, device="cuda:0", expert_dtype="bf16")
tokenizer = load_diffusion_gemma_tokenizer(path)
decoder = EntropyBoundParallelDecoder(model.config.vocab_size, seed=42)
cache = CanvasPolestarCache(len(model.layers), topk=2, refresh_every=8)
llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                         canvas_cache=cache)
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "What is 12 times 8?"}],
                                       add_generation_prompt=True, tokenize=True)
prompt_ids = torch.tensor([prompt], device="cuda:0")
output_ids = llm.generate(prompt_ids, gen_length=512)
print(tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True))
```

Omit `canvas_cache` for plain dInfer decoding. The single-GPU example needs about 50 GB of GPU memory with
bf16 experts. For several GPUs, start one process per GPU in an initialized `torch.distributed` group, give
each its own `device`, and run the same prompts on every rank; the benchmark below does this.

### Check GSM8K with DiffusionGemma

`evaluate_gsm8k_diffusion_gemma.py` uses the tasks and few-shot counts of
[`configs/evaluation/text.json`](../../configs/evaluation/text.json), the settings in
[`configs/diffusiongemma26b.json`](configs/diffusiongemma26b.json), and the same prompt and stop-sequence
handling as the LLaDA evaluator. `--gpus` lists the GPUs to split the experts across.

```bash
python integrations/dinfer/benchmarks/evaluate_gsm8k_diffusion_gemma.py \
  --backend dinfer --task gsm8k --limit 2 --seed 42 --gpus 0,1 --expert-dtype int8 \
  --output-path results/dgemma-dinfer-smoke

python integrations/dinfer/benchmarks/evaluate_gsm8k_diffusion_gemma.py \
  --backend polestar --task gsm8k --limit 2 --seed 42 --gpus 0,1 --expert-dtype int8 \
  --output-path results/dgemma-polestar-smoke
```

`--polestar-topk` and `--polestar-refresh-every` override the preset. Each run writes `results.json`,
`samples_gsm8k.jsonl`, and `serving_metrics.json`. The last records tokens per second, seconds per
denoising step, and the fraction of canvas tokens recomputed on partial steps.

**Measured results** (first 50 GSM8K test questions, 5-shot, seed 42, int8 experts split across 2× NVIDIA TITAN RTX
24 GB, 512-token budget, one run each):

| Backend | Accuracy (strict / flexible) | Denoising steps | ms per step | Tokens per second |
|---|---|---|---|---|
| dInfer | 0.98 / 0.98 | 433 | 506 | 32.3 |
| dInfer + Polestar-Cache (`topk=2`, `refresh_every=8`) | 0.98 / 0.98 | 436 | 489 | 33.4 |

Both runs miss the same question. Polestar-Cache recomputed 66% of the canvas tokens on partial steps and
reduced the time per step by 3%. On these GPUs, a step's cost is dominated by work that does not depend on how
many tokens are recomputed (expert dequantization, weight reads), and the uniform-noise sampler re-noises most
canvas positions in early steps, which limits reuse. These are single runs; repeat them before relying on
differences of a few percent.

### Polestar-Commit for DiffusionGemma

LLaDA's Polestar-Commit commits masked tokens early. DiffusionGemma has no mask token: every canvas position
is
re-predicted on every step, and the entropy-bound sampler keeps (accepts) only the most confident positions
for
the next step. `PolestarCommitEntropyDecoder` in
[`dinfer_canvas_commit.py`](../../src/polestar/integrations/dinfer_canvas_commit.py) ports each part of
LLaDA's Commit that has a canvas equivalent. Each part is off by default:

| LLaDA Polestar-Commit | DiffusionGemma port | Benchmark flag |
|---|---|---|
| Permanent commit at confidence ≥ τ | Freeze: once its raw confidence reaches τ, a position keeps its argmax token for the rest of the canvas | `--commit-freeze-threshold` |
| Drift commit (0.7 < confidence < 0.9 and a final-layer attention drift jump) | The position is also accepted on that step | `--commit accept` |
| Commits trigger Polestar-Cache refreshes | Full canvas step after more than N newly frozen positions (`CanvasPolestarCache(refresh_after_commits=N)`) | `--polestar-refresh-after-commits` |
| Commit at least one token per step | Not needed: the entropy bound accepts at least one position per step | – |
| Suffix-block commit | No equivalent: a denoising pass never sees the next canvas | – |

`--commit record` computes the drift rule without changing any decision. With `--commit-trace-dir`, it writes
per-step, per-position traces. [`analyze_dgemma_commit_trace.py`](benchmarks/analyze_dgemma_commit_trace.py)
uses these traces to test whether drift identifies tokens that are already final, and to simulate the freeze.
Its other settings (`--commit-threshold`, `--commit-lower-bound`, `--commit-window`, `--commit-gate-scale`,
`--commit-drift-cap`, `--commit-confidence`) override the preset's `commit` block.

```bash
python integrations/dinfer/benchmarks/evaluate_gsm8k_diffusion_gemma.py \
  --backend polestar --commit record --commit-trace-dir results/dgemma-commit-trace/traces \
  --task gsm8k --limit 50 --seed 42 --gpus 0,1 --expert-dtype int8 --output-path results/dgemma-commit-trace

python integrations/dinfer/benchmarks/analyze_dgemma_commit_trace.py \
  --trace-dir results/dgemma-commit-trace/traces --output results/dgemma-commit-trace/REPORT.md

python integrations/dinfer/benchmarks/evaluate_gsm8k_diffusion_gemma.py \
  --backend polestar --commit accept --commit-freeze-threshold 0.999 --polestar-refresh-after-commits 24 \
  --task gsm8k --limit 50 --seed 42 --gpus 0,1 --expert-dtype int8 --output-path results/dgemma-full-commit
```

**Measured results.** Same setup as the GSM8K results above.

**Trace analysis (record mode).**
- At equal confidence, positions with a drift event are no more likely to be final than positions without one.
  Examples: 58.8% vs 64.5% for confidence in [0.7, 0.8), and 70.1% vs 78.5% in [0.8, 0.9) (record run without
  Cache). This holds across all 54 tested drift settings.
- The entropy-bound sampler already accepts about 147 of 256 positions per step, and 99.1–99.6% of the
  accepted tokens are final.
- Freezing at LLaDA's τ = 0.9 would freeze a wrong token 6–7.5% of the time; at τ = 0.999 the rate is
  0.2–0.3%.

**Full port** (Polestar-Cache on in every run; τ = 0.999, N = 24, which is LLaDA's 3 per 32-token block scaled
to the 256-token canvas):

| Seed | Run | Accuracy | Denoising steps | ms per step | Tokens per second |
|---|---|---|---|---|---|
| 42 | Polestar-Cache | 0.98 | 436 | 489 | 33.4 |
| 42 | + full Commit | 0.94 | 382 | 537 | 34.5 |
| 42 | + freeze and refresh on commits (no drift) | 0.96 | 430 | 510 | 33.0 |
| 42 | + freeze only | 0.96 | 460 | 477 | 31.7 |
| 43 | Polestar-Cache | 0.90 | 459 | 481 | 31.6 |
| 43 | + full Commit | 0.94 | 456 | 509 | 30.5 |

On DiffusionGemma, Polestar-Commit runs correctly but gives no reliable speedup.
- **Steps don't drop reliably.** Step counts vary from 382 to 460 between equivalent runs, and the reduction
  at seed 42 did not repeat at seed 43.
- **Refreshing on commits costs 4–10% per step.** Most freezes happen in a canvas's first one or two steps, so
  the refreshes replace Polestar-Cache's partial steps with full ones.
- **Accuracy stays within run-to-run noise:** 0.90–0.98 across all runs.

The entropy-bound sampler already keeps confident tokens. On a canvas that is partly re-noised every step,
attention drift mostly reflects that noise rather than marking settled tokens.
