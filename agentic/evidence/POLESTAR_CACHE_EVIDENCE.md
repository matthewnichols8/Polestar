# Polestar-Cache in DiffusionGemma (dInfer): how it's integrated, and evidence it runs

**Short answer:**
- DiffusionGemma, inside the dInfer framework, calls Polestar-Cache on **every layer of every denoising step**.
- On most steps the cache **recomputes only part of the canvas**: it skipped 2–75% of tokens in this trace and 34% on average on GSM8K. It materializes **only the experts those tokens use** (35–40 of 64 per GPU).
- GSM8K accuracy was unchanged (49/50 with and without it).
- The wall-clock gain on our TITAN RTX GPUs is small: about 3% per step on GSM8K.

The evidence comes from:
1. the code path;
2. CPU unit tests;
3. a fresh instrumented GPU trace (`trace_polestar_cache.py` → `trace.json`, `trace.log`);
4. the 50-question GSM8K benchmark.

All code is in the Polestar fork at `/scratch/mi72/Polestar`. It was added on `feature/diffusion-gemma` (the cache in commit `3aa43bb`, pushed to `github.com/matthewnichols8/Polestar`); the fork is currently checked out on `feature/dgemma-prefix-reuse`, which builds on it.

---

## 1. Where the pieces live

| Layer | File | Role |
|---|---|---|
| Polestar | `src/polestar/integrations/dinfer_canvas_cache.py` | `CanvasPolestarCache`: chooses which tokens to recompute and reuses the rest |
| Polestar (reused, unchanged) | `src/polestar/models/llada/clustering.py`, `system_cache.py` | `k_means`, `compute_cluster_drift_kl`, `incremental_centroids`: the same clustering and drift math as Polestar's LLaDA cache |
| dInfer | `integrations/dinfer/python/dinfer/decoding/generate_uniform.py` | `CanvasDiffusionLLM(..., canvas_cache=cache)` resets the cache per canvas and hands it to the model |
| dInfer model | `integrations/dinfer/python/dinfer/model/modeling_diffusion_gemma.py` | `denoise(..., layer_runner=cache)` runs every layer *through* the cache |

**The model has no Polestar dependency.** It exposes one hook, `layer_runner`. With `layer_runner=None` (plain dInfer), the original code path runs unchanged, and that is tested bit for bit.

## 2. How a denoising step flows through the cache

```
CanvasDiffusionLLM.generate                          (generate_uniform.py)
 └─ for each 256-token canvas:
     ├─ canvas_cache.reset()                          new canvas → forget states
     └─ for each denoising step:
         CanvasDiffusionIteration.forward
          └─ model.denoise(canvas, kv, sc, layer_runner=canvas_cache)      (generate_uniform.py:269)
              ├─ layer_runner.begin_step(canvas)        decide full vs partial step, mark tokens whose input changed
              ├─ for each of the 30 layers:
              │    h = layer_runner.run_layer(layer, h, ...)               (modeling_diffusion_gemma.py:545)
              │      ├─ full step    → _run_full: normal layer + save h_in/h_out + k-means clusters
              │      └─ partial step → _run_partial:
              │            K/V for all tokens (so attention sees current states)
              │            select = changed tokens ∪ tokens in the top-k clusters whose centroid
              │                     proxy-attention drifted most (KL vs last step)
              │            attention + FFN/MoE for the selected tokens only, with sparse expert dequant
              │            other tokens: reuse cached update  out = s·h_now + (out_old − s·h_old)
              └─ layer_runner.end_step()                record the recomputed fraction
```

**When a step is "full":**
- the first step of every canvas;
- every `refresh_every` steps (8 in our preset).

Every other step is "partial". Our preset (`integrations/dinfer/configs/diffusiongemma26b.json`) uses `clusters=8, topk=2, refresh_every=8, reuse="delta"`.

**What changed from Polestar's LLaDA cache:**
- **Where the reuse happens:** DiffusionGemma's prompt cache comes from a causal encode and never drifts, so there's nothing to refresh there. The savings have to come from the canvas *across denoising steps*.
- **Which tokens are selected:** DiffusionGemma re-predicts every canvas token on every step. So, besides the drifting clusters, any token whose input changed must also be recomputed.
- **Expert loading:** int8 experts are dequantized on use. Partial steps dequantize only the experts the selected tokens route to.
- **Multi-GPU agreement:** the token selection is made on rank 0 and broadcast, so both GPUs process the same tokens.

## 3. Evidence from the unit tests (CPU, tiny random DiffusionGemma)

`tests/test_dinfer_canvas_cache.py`: **13 passed** (re-run for this write-up). What they establish:

| Test | Shows |
|---|---|
| `test_full_step_equals_the_normal_pass` | A full step through the cache equals the plain layer: no hidden change of results |
| `test_partial_step_with_every_cluster_selected_matches_full_recompute` | The partial path is correct: selecting every token reproduces a full recompute (atol 1e-5) |
| `test_partial_step_reuses_unselected_tokens` | Unselected tokens really are reused, not recomputed |
| `test_selection_is_dirty_tokens_plus_the_top_clusters` | The selection is exactly the changed tokens plus the `topk` highest-drift clusters |
| `test_refresh_schedule` | Full steps happen at step 0 and every `refresh_every` steps |
| `test_reuse_modes` | `reuse="delta"` and `reuse="state"` both behave as specified |
| `test_sparse_dequant_matches_the_full_expert_pass` | Dequantizing only the used experts gives the same expert output |
| `test_same_seed_gives_the_same_clusters` | Clustering is deterministic given `polestar_seed` |
| `test_canvas_llm_with_the_cache` | End to end through `CanvasDiffusionLLM` with the cache attached |
| `test_invalid_settings` | Bad settings are rejected |

A separate test, in `test_diffusion_gemma_dinfer.py`, checks that `layer_runner=None` is bitwise identical to the original path.

## 4. Evidence from the real model: an instrumented GPU trace

`trace_polestar_cache.py` wraps the cache's `_run_full` / `_run_partial`, the layer FFN (`Layer._ffn`), expert dequantization (`Experts._weights` / `_weights_subset`) and `denoise` timing. It doesn't edit any library code. It runs three prompts with and without the cache:
- `google/diffusiongemma-26B-A4B-it`;
- int8 experts split across 2× TITAN RTX (64 experts per GPU);
- decoder seed 42, `topk=2`, `refresh_every=8`.

Raw data: `trace.json`; console output: `trace.log`.

### 4a. The cache is called for every layer of every step

| Prompt | Denoising steps (cache) | `_run_full` calls | `_run_partial` calls | = steps × 30 layers |
|---|---|---|---|---|
| Janet's ducks (GSM8K) | 6 (1 full + 5 partial) | 30 | 150 | 180 ✓ |
| Robe bolts (GSM8K) | 7 (1 full + 6 partial) | 30 | 180 | 210 ✓ |
| Fibonacci function | 9 (2 full + 7 partial) | 60 | 210 | 270 ✓ |

Without the cache, both counts are 0 and every step runs the plain layers.

### 4b. Partial steps process fewer tokens

Mean canvas tokens per layer through the FFN/MoE during denoising (out of 256):

| Prompt | No cache | Cache, full steps | Cache, partial steps | Share of the canvas recomputed, per partial step |
|---|---|---|---|---|
| Janet's ducks | 256 | 256 | **167** | 0.83 → 0.62 → 0.47 → 0.34 |
| Robe bolts | 256 | 256 | **121** | 0.59 → 0.43 → 0.31 → 0.25 → 0.25 |
| Fibonacci | 256 | 256 | **219** | 0.98, 0.97, 0.97, 0.93, 0.64, 0.49 |

Each run's list starts with the 1.0 of its first (full) step, which the table omits. For example, for Janet's ducks the cache reports `[1.0, 0.83, 0.62, 0.47, 0.34]`.

The recomputed share falls as the canvas settles. Early steps recompute almost everything, because DiffusionGemma's uniform-noise sampler re-noises most positions early. The Fibonacci answer changed more from step to step, so it reused less.

### 4c. Partial steps dequantize only the experts they need

| Prompt | Full expert dequants during denoising (cache) | Sparse dequants | Experts materialized per sparse call (of 64 per GPU) |
|---|---|---|---|
| Janet's ducks | 30 (the full step only) | 150 | **39.1** |
| Robe bolts | 30 | 180 | **35.4** |
| Fibonacci | 60 | 210 | **40.5** |

No partial step ever dequantized the full set of experts. Every partial-step layer took the sparse path.

### 4d. Time per step

| Prompt | No cache (ms/step) | Cache: full step | Cache: partial step |
|---|---|---|---|
| Janet's ducks | 308 | 375 | **306** |
| Robe bolts | 311 | 327 | **279** |
| Fibonacci | 310 | 325 | **315** |

**Reading the timings:**
- Partial steps are 1–10% cheaper than plain steps.
- Full steps cost 5–20% more, because they also run k-means and store states.
- The savings are small because on these GPUs much of a step's cost doesn't shrink with fewer tokens: per-layer kernel launches, the cross-GPU all-reduce, attention K/V for the whole canvas, and reading the weights.

### 4e. Outputs

**The cache is approximate,** so the text isn't identical to the uncached run (`same_text_as_no_cache: false` for all three). The answers are the same:

| Prompt | Without the cache | With the cache |
|---|---|---|
| Janet's ducks | **$18** | **$18** |
| Robe bolts | **3 bolts** | **3 bolts** |
| Fibonacci | an equivalent iterative function with an explanation | an equivalent iterative function with an explanation |

The cached runs also took fewer steps here (6 vs 9, 7 vs 8, 9 vs 17). That's because the outputs differed, not a reliable effect of the cache: over 50 GSM8K questions the step counts were equal (436 vs 433).

## 5. Evidence at benchmark scale: GSM8K, 50 questions

From `integrations/dinfer/benchmarks/evaluate_gsm8k_diffusion_gemma.py`:
- lm-eval, 5-shot, seed 42;
- int8 experts on 2× TITAN RTX;
- results in `/scratch/mi72/polestar_runs/gsm8k50-{dinfer,polestar}/`.

| Backend | Accuracy (strict / flexible) | Steps | ms / step | Tokens / s | Mean share of the canvas recomputed on partial steps |
|---|---|---|---|---|---|
| dInfer | 0.98 / 0.98 | 433 | 506 | 32.3 | n/a (cache off) |
| dInfer + Polestar-Cache | 0.98 / 0.98 | 436 | **489** | **33.4** | **0.66** |

`mean_recompute_frac_partial_steps` comes from the cache's own `stats()`, reported for every request in `serving_metrics.json`. It only exists when the cache ran. Both runs miss the same question.

## 6. Conclusion and limits

**The integration is real:**
- dInfer's canvas loop routes every layer of every denoising step through `CanvasPolestarCache`.
- On partial steps the cache skips about a third of the canvas tokens (2–75% in the trace) and about 40% of the experts.
- The unit tests show the partial computation is exact when every token is selected, and that unselected tokens are reused.

**The limits are just as clear:**
- On TITAN RTX the time saving is small: about 3% per step on GSM8K, 1–10% on partial steps in the trace. The work that shrinks with the cache is only part of each step.
- Early denoising steps change most tokens, which limits reuse.
- These are single runs. Repeat them before relying on differences of a few percent.

The cache should help more where per-token compute dominates: longer contexts, faster GPUs with less launch and communication overhead, or larger `refresh_every` and smaller `topk` (at some accuracy risk).

## Reproduce
```bash
cd /scratch/mi72/agentic-dgemma/evidence
PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp /scratch/mi72/envs/polestar/bin/python trace_polestar_cache.py --gpus 0,1 --out trace.json
```
```bash
cd /scratch/mi72/Polestar && CUDA_VISIBLE_DEVICES= PYTHONPATH= /scratch/mi72/envs/polestar/bin/python -m pytest -q tests/test_dinfer_canvas_cache.py
```
