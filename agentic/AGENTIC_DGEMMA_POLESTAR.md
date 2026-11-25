# Running DiffusionGemma with Polestar CACHE on agentic benchmarks

**What this covers:** how `google/diffusiongemma-26B-A4B-it` went from a diffusion LLM with no serving stack to a multi-turn tool-using agent on τ²-bench, with **Polestar CACHE** on the denoising path.
- Polestar COMMIT is not integrated yet.
- **Detailed results:** `results/PHASE0_RESULTS.md`, `results/PHASE1_RESULTS.md`, `evidence/POLESTAR_CACHE_EVIDENCE.md` and `results/tau2/TAU2_RETAIL_RESULTS.md`.

## The stack

```
τ²-bench (retail)  ──LiteLLM──►  serve_openai.py  (OpenAI /v1/chat/completions, port 8000)
   │                               │  gemma_format.py: OpenAI messages+tools  ⇄  Gemma chat template / tool syntax
   │ user simulator: GPT-4.1       │  prefix reuse: keep the conversation's KV cache between turns
   ▼                               ▼
 OpenAI API                 CanvasDiffusionLLM (dInfer, Polestar fork)
                              │  causal encode of the new prompt tokens → per-layer KV cache
                              │  denoise 256-token canvases with EntropyBoundParallelDecoder
                              ▼
                     DiffusionGemma, expert-parallel on 2× TITAN RTX (int8 experts)
                              │  every layer of every denoising step runs through ↓
                              ▼
                     CanvasPolestarCache  (Polestar CACHE: partial recompute + sparse expert dequant)
```

| Piece | Where |
|---|---|
| Model, decoder, canvas loop, prefix reuse | `/scratch/mi72/Polestar/integrations/dinfer/python/dinfer/` (branches `feature/diffusion-gemma`, `feature/dgemma-prefix-reuse`) |
| Polestar CACHE | `/scratch/mi72/Polestar/src/polestar/integrations/dinfer_canvas_cache.py` |
| Server and format translation | `/scratch/mi72/agentic-dgemma/server/` (`serve_openai.py`, `gemma_format.py`) |
| Probes | `/scratch/mi72/agentic-dgemma/probes/` |
| τ²-bench | `/scratch/mi72/tau2-bench` (commit `19f13ea`) |

## Step 1. DiffusionGemma in dInfer
DiffusionGemma isn't an autoregressive model:
- It **encodes** the prompt causally into a per-layer KV cache (sliding-window layers keep only their window).
- It then **denoises** a 256-token canvas bidirectionally over several steps. Each step re-predicts every canvas token.

We added the model, an entropy-bounded parallel decoder and a canvas loop (`CanvasDiffusionLLM`) to the dInfer copy bundled in the Polestar repo:
- **Experts:** int8, split across both GPUs (64 per GPU), with an all-reduce after each MoE layer.
- **The model has no Polestar dependency.** It exposes one hook, `denoise(..., layer_runner=...)`, and with `None` the plain path runs unchanged.
- **Correctness:** CPU tests cover the model, decoder, cache and loop, including bit-for-bit checks of the plain path.

## Step 2. Polestar CACHE on the denoising path
`CanvasPolestarCache` is the `layer_runner`, so **every layer of every denoising step goes through it**.
- **Full steps:** step 0 of each canvas, and every 8 steps after that. They run the normal layer, store each layer's input and output states, and cluster the canvas tokens with k-means.
- **Partial steps:** recompute only the tokens whose input changed, plus the tokens in the 2 clusters whose attention drifted most. The other tokens reuse their cached update.
- **Sparse dequantization:** only the int8 experts that the recomputed tokens route to are dequantized.
- **Keeping the two GPUs in step:** the token selection is made on rank 0 and broadcast to rank 1.

**Why the reuse happens on the canvas, unlike Polestar's LLaDA cache:** DiffusionGemma's prompt KV comes from a causal encode and never changes, so there's nothing to refresh there. The redundancy is between consecutive denoising steps.

**Evidence** (`evidence/POLESTAR_CACHE_EVIDENCE.md`):
- 13 unit tests, e.g. a partial step that selects every token equals a full recompute.
- A GPU trace: the cache is called on steps × 30 layers, partial steps recompute 25–98% of the canvas, and 35–40 of 64 experts are dequantized.
- GSM8K: 0.98 accuracy with and without the cache, 3% faster per step.

## Step 3. Checking that the model can call tools at all (Phase 0)
DiffusionGemma's chat template supports tools natively, using Gemma's own syntax rather than JSON:
```
<|tool_call>call:get_order_details{order_id:<|"|>#W6390527<|"|>}<tool_call|>
```
Tool results are placed inside the model's turn, right after the call.

**Why check:** a diffusion model decodes a whole call in parallel, not left to right, so structured output was the main risk.

**Probes:** 225 generations across 7 levels, from single calls up to multi-turn conversations and policy checks.
- **0 malformed calls**; 216–219 of 225 correct.
- **Main error:** an occasionally dropped character in long IDs.
- **Decision:** go ahead without constrained decoding.

## Step 4. An OpenAI-compatible server (Phase 1)
τ²-bench talks to models through LiteLLM's OpenAI API, so `serve_openai.py` exposes `/v1/chat/completions` over the expert-parallel model.

**Format translation** (`gemma_format.py`):
- OpenAI `messages` + `tools` → Gemma chat template:
  - JSON-string `arguments` become mappings;
  - `tool` messages are matched to their call by `tool_call_id`;
  - `developer` becomes `system`.
- Generated text → OpenAI response:
  - output is cut at the model's first self-written `<|tool_response>`, or at end of turn;
  - Gemma calls are parsed into `tool_calls` with JSON arguments, and `finish_reason` is set to `tool_calls`;
  - a call that won't parse is returned as plain text rather than a broken call.

**Serving two GPUs:**
- Rank 0 runs the HTTP server and broadcasts each prompt; rank 1 follows.
- Requests are processed one at a time.
- Every response carries a `dgemma` diagnostics field: tokens, steps, reused prompt tokens, malformed calls, and the Polestar CACHE stats when it's on. Each response is also logged to `requests.jsonl`.

**Prefix reuse across turns:** each agent turn's prompt extends the previous one, and the encode is causal.
- So the server keeps the last few conversations' KV caches and encodes only the new tokens.
- Reuse is bitwise equal to encoding the prompt in two parts, which is tested.
- At 12k tokens a turn takes **11.8 s instead of 26.1 s**. In τ²-bench about **87%** of prompt tokens were reused.

**Phase 1 checks:** 45/45 probes through HTTP; a 9-turn scripted conversation; a LiteLLM tool round trip.

## Step 5. τ²-bench with Polestar CACHE (Phase 2)
**τ²-bench retail:** the agent follows a store policy and uses tools to authenticate customers, look up orders, and make returns, exchanges and changes.
- A **GPT-4.1 user simulator** plays the customer.
- Scoring checks the final database state and the required outputs.

**The server, with CACHE on:**
```bash
cd /scratch/mi72/agentic-dgemma/server
PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp /scratch/mi72/envs/polestar/bin/python serve_openai.py \
  --gpus 0,1 --port 8000 --polestar --polestar-topk 2 --polestar-refresh-every 8 --log-dir <results dir>/server
```
**The benchmark** (agent = our server, user = GPT-4.1):
```bash
cd /scratch/mi72/tau2-bench && set -a && . /scratch/mi72/.tau2.env && set +a
/scratch/mi72/envs/tau2/bin/tau2 run --domain retail --task-split-name base --num-tasks 25 --num-trials 1 \
  --agent-llm openai/diffusiongemma \
  --agent-llm-args '{"api_base":"http://127.0.0.1:8000/v1","api_key":"x","max_tokens":512}' \
  --user-llm gpt-4.1 --max-concurrency 1 --save-to dgemma-polestar-cache-retail25
```

**Evidence that CACHE was on:** all 293 agent requests in the CACHE run log `polestar_cache: true`, with per-canvas stats.
- 407 full steps and 1,637 partial steps;
- 65% of the canvas recomputed on average on partial steps.

**Results (25 retail tasks, 1 trial):**

| | dInfer baseline | Polestar CACHE |
|---|---|---|
| Solved (pass^1) | 16/25 (0.64) | **20/25 (0.80)** |
| ms per denoising step | 601 | **568 (−5.5%)** |
| Malformed calls / empty replies | 0 / 1 | 1 / 0 |

- **Accuracy:** the difference isn't statistically significant (p = 0.22). CACHE did not hurt agentic accuracy.
- **Example of a solved task:** authenticate → look up orders (in parallel) → check product options → confirm → `return_delivered_order_items`.

## Known issues and next steps
- **Server hang with CACHE on.** It hung twice in about 10–12k-token conversations: the two GPUs fell out of step. The CACHE run was restarted twice and the unfinished tasks were rerun.
  - **Suspect:** a per-rank decision (accepted tokens or stop) that isn't broadcast.
  - **Fix:** broadcast the decoder's decisions from rank 0, or add a per-step consistency check.
  - Needed before longer runs or COMMIT.
- **Polestar COMMIT:** integrate it, then run `dgemma-polestar-cache-commit-retail25` with the same setup.
- **Scale up:** all 50 retail tasks, airline, and `--num-trials 4` for pass^k.
- **SWE-bench:** blocked until Docker or apptainer is available on the server.
