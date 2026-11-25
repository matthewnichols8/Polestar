# τ²-bench retail: DiffusionGemma, dInfer baseline vs Polestar CACHE vs full Polestar (CACHE + COMMIT)

## Update (2026-11-23): full Polestar, CACHE + COMMIT, on tasks 0–19

**Short answer**
- **Full Polestar solved 18/20**, against 16/20 for CACHE only and 13/20 for the dInfer baseline, on the same 20 tasks.
  - The gain over the baseline is close to significant: 5 tasks gained, 0 lost, exact McNemar p = 0.06.
  - Over CACHE only it isn't: 2 gained, 0 lost, p = 0.5.
- **Full Polestar was not faster.**
  - It took 7.5% fewer denoising steps per reply than the baseline, but each step was slower: 618 ms vs 601 ms (baseline) and 565 ms (CACHE only).
  - The reason: COMMIT's freezes trigger CACHE refreshes so often (`refresh_after_commits=24`) that 65% of steps were full recomputes, against 20% with CACHE alone.
  - Per reply it came to 3.95 s, against 4.16 s for the baseline and 3.89 s for CACHE only.
- **Tool calls:** no malformed calls. There was 1 empty reply (task 4); τ²-bench retried it and the retry passed.

| Tasks 0–19 | dInfer baseline | Polestar CACHE | **Full Polestar (CACHE + COMMIT)** |
|---|---|---|---|
| Solved / pass^1 | 13/20 = 0.65 | 16/20 = 0.80 | **18/20 = 0.90** |
| Per task (1 = solved) | `11000111110111011100` | `11000111111110111111` | `11001111111111111111` |
| Malformed calls | 0 | 0 | 0 |
| Empty replies (τ²-bench retries) | 1 (task 4) | 0 | 1 (task 4, passed on retry) |
| Mean new tokens per reply | 85.7 | 83.3 | 77.0 |
| Denoising steps per reply | 6.92 | 6.88 | **6.40** |
| ms per step (prefill included) | 601 | **565** | 618 |
| Server time per reply | 4.16 s | **3.89 s** | 3.95 s |
| CACHE full / partial steps | n/a | 312 / 1,244 | 925 / 489 |
| CACHE refreshes triggered by COMMIT | n/a | n/a | 638 |
| COMMIT: drift-gated acceptances / tokens frozen | n/a | n/a | 4,242 / 55,562 |
| GPT-4.1 cost | $0.27 | $0.27 | $0.27 |

**What the speed rows cover:**
- The baseline and CACHE columns come from all logged requests of those runs: the baseline's 25 tasks, and CACHE part 1, which covers tasks 0–19 plus task 20 up to the hang.
- The COMMIT column covers tasks 0–19 only.
- Per-step and per-reply averages are comparable across columns; the totals are not.

**Settings for the full Polestar run:**
- **Label:** `dgemma-polestar-cache-commit-retail20`.
- **Code:** Polestar `feature/dgemma-polestar` @ `6c124e6`. The full-COMMIT values come from the preset's uncommitted edits, and are passed explicitly to the server.
- **CACHE:** topk 2, refresh every 8, `refresh_after_commits` 24.
- **COMMIT:** mode `accept`; threshold 0.9; lower bound 0.7; window 3; gate scale 10; drift cap 2; freeze at 0.999.
- **Server:** `serve_openai.py --polestar --polestar-commit --cache-slots 2`, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The command is in `config.json`.

**What happened during the run:**
- **First attempt: out of memory.** GPU 1 ran out of memory at a ~9.5k-token prompt. COMMIT keeps the final layer's attention over the whole prompt for 3 steps, and 1.6 GB was fragmented.
  - That attempt was stopped after the crash, and its logs are kept in `attempt1-oom/`.
  - The rerun used expandable segments and 2 cache slots instead of 4. Neither setting changes what the model generates.
- **The earlier hang didn't come back.** The new decoder broadcasts rank 0's canvas, accepted tokens and stop flag, which is the fix suggested in §5 below.

**Next:**
- Tune `refresh_after_commits` (e.g. 64–128, or off) so COMMIT doesn't cancel CACHE's savings. Then rerun.
- Use more tasks and trials before claiming an accuracy effect. Each configuration has one trial on 20 tasks.

---

# Original report (2026-11-04): 25 tasks, dInfer baseline vs Polestar CACHE

**Short answer**
- **Polestar CACHE** solved **20/25 (pass^1 0.80)**. The **dInfer baseline** (no Polestar) solved **16/25 (pass^1 0.64)**.
  - With 25 tasks and one trial each, the gap isn't statistically significant: the runs disagree on 6 tasks, 5 of them in CACHE's favour, exact McNemar p = 0.22.
  - Read it as "the cache didn't hurt agentic accuracy", not as "the cache improves it".
- **Per denoising step, CACHE was 5.5% faster** (568 vs 601 ms, prefill included). Partial steps recomputed 65% of the canvas on average.
- **Tool calls were reliable in both runs:**
  - 1 malformed call in 584 agent replies (in the CACHE run);
  - 1 empty reply (in the baseline), which τ²-bench retried;
  - no self-written tool responses.
- **Problem found: the server hung twice with CACHE on, and never without it.** In both cases the two GPUs fell out of step late in a long conversation (~10–12k tokens). I restarted the server and reran the unfinished tasks (see §5). This must be fixed before longer runs, and before adding Polestar COMMIT.

## 1. Run labels

| Label | Polestar CACHE | Polestar COMMIT | Status |
|---|---|---|---|
| `dgemma-dinfer-retail-smoke2` | off | off | 2-task test, 1/2 solved |
| `dgemma-dinfer-retail25` | off | off | ✅ this report |
| `dgemma-polestar-cache-retail25` | **on** (topk 2, refresh every 8, 8 clusters, delta reuse, seed 42) | off | ✅ this report |
| `dgemma-polestar-cache-commit-retail25` | on | on | planned, once COMMIT is integrated |

- **Per-run config:** each run's folder in `results/tau2/` has a `config.json` with the exact server flags, the Polestar repo commit (`9af281c`, branch `feature/dgemma-prefix-reuse`) and the τ²-bench settings.
- **Raw τ²-bench output:** `/scratch/mi72/tau2-bench/data/simulations/<label>[-partN]/results.json`.

## 2. Setup

| | |
|---|---|
| Agent model | `google/diffusiongemma-26B-A4B-it` on dInfer (bundled in the Polestar fork); int8 experts, expert-parallel over 2× TITAN RTX |
| Decoding | entropy bound 0.1, ≤ 48 denoising steps per 256-token canvas, thinking off, `max_tokens` 512 |
| Serving | `agentic-dgemma/server/serve_openai.py`: OpenAI-compatible, Gemma tool-call translation, per-turn prefix KV reuse |
| Benchmark | τ²-bench `19f13ea`; domain `retail`, split `base`, tasks 0–24, 1 trial, concurrency 1 |
| User simulator | `gpt-4.1` (OpenAI) |
| Same in both runs | everything except `--polestar --polestar-topk 2 --polestar-refresh-every 8` |

- **Server change between the runs (logging only):** before the CACHE run I added the cache's own stats (`polestar_cache`, `canvas_cache`) to the per-request diagnostics. It doesn't change generation.
- **Evidence the cache ran:** all 293 CACHE-run requests in `requests.jsonl` carry `polestar_cache: true` and per-canvas full/partial step counts. In the baseline those fields are absent.

## 3. Results

| | dInfer baseline | Polestar CACHE |
|---|---|---|
| **Solved / pass^1** | 16/25 = **0.64** | 20/25 = **0.80** |
| Agent replies (to τ²-bench) | 288 | 295 |
| of which tool calls | 122 | 127 |
| Malformed tool calls | 0 | 1 |
| Empty replies (τ²-bench retries) | 1 (task 4, retried and passed) | 0 |
| Self-written tool responses | 0 | 0 |
| Mean messages per task | 25.2 | 25.9 |
| Mean prompt / max prompt (tokens) | 6,310 / 12,745 | 6,515 / 13,041 |
| Prompt tokens reused from the prefix cache | 86.1% | 86.9% |
| Denoising steps | 2,015 | 2,044 |
| **ms per step** (prefill included) | 601 | **568 (−5.5%)** |
| Mean server time per reply | 4.16 s | 3.96 s |
| Mean time per task (τ²-bench `duration`) | 49.8 s | 49.5 s |
| Cache: full / partial steps | n/a | 407 / 1,637 |
| Cache: mean share of the canvas recomputed on partial steps | n/a | 0.65 |
| GPT-4.1 user-simulator cost | $0.34 | $0.34 |

**Per task** (1 = solved; tasks 0–24, left to right):
```
dInfer baseline   1100011111011101110001011
Polestar CACHE    1100011111111011111101111
```
- **Disagreements:** tasks 10, 14, 18, 19 and 22 were solved only with CACHE; task 13 only without it.
- **Why the runs can't be compared token for token:** the GPT-4.1 customer is not deterministic, and the cache changes the agent's text. So each disagreement is a different conversation, not the same conversation with one different decision.

**Speed, read with care:**
- The 5.5% per-step gain is consistent with GSM8K (about 3%) and the trace (1–10% on partial steps).
- Per-step time includes prefill, which the cache doesn't touch. Prefix reuse keeps prefill small, since about 87% of prompt tokens are reused.
- Per-step denoising cost still grows with context. Each canvas attends over the whole 6–13k-token prompt, and the cache saves FFN/MoE work only for the canvas tokens it skips.

## 4. Failure modes seen (both runs)
- **Malformed call (1, CACHE):** the parser couldn't convert one call. The server returns such a call as plain text rather than a broken call, as designed.
- **Empty reply (1, baseline):** the reply had no text and no tool call. τ²-bench raised an error and restarted the task (retry 1/3).
- Detailed per-task failure analysis (wrong item, wrong payment method, skipped confirmation, …) is not done yet. The conversations are in the `results.json` files.

## 5. The CACHE-run server hangs (must fix)

**What happened:**
| Part | Tasks completed | Hung at | Last reply | Prompt at hang |
|---|---|---|---|---|
| 1 | 0–19 | task 20 | 19:18:12 UTC | ~11.7k tokens |
| 2 (restart) | 20–22 | task 23 | 19:53:58 UTC | ~10.2k tokens |
| 3 (restart) | 23–24 | (none) | | |

- **Unfinished attempts don't count:** the two hung attempts never finished, so they aren't in any results file. Tasks 20 and 23 were each rerun from scratch once, which is equivalent to a first attempt. τ²-bench retries for crashes are a separate mechanism, and the baseline used one (task 4).
- **Symptoms:**
  - `/health` still answered, but chat requests got no reply.
  - GPU 0 sat idle and its process was asleep. GPU 1's process was at 100% CPU and 100% GPU, waiting in a NCCL collective.
  - No error or traceback appeared.
- **Probable cause (unconfirmed):** the two expert-parallel ranks disagreed on some decision and stopped issuing the same sequence of collectives. Rank 0 finished and went back to waiting for a request; rank 1 entered a collective that rank 0 never joined.
  - The cache's token selection is broadcast from rank 0, so it can't diverge.
  - Candidates are decisions each rank makes **on its own**, e.g. the decoder's accepted tokens or end-of-canvas/EOS stop, computed on each rank from its own logits. Any tiny numerical difference between ranks could flip such a decision. Why only the cache path diverges isn't known yet.
- **Couldn't confirm:** `py-spy` cannot attach (ptrace is restricted for unprivileged users on this machine).
- **Logs:** `dgemma-polestar-cache-retail25/server-part{1,2}-hung*`.
- **Suggested fixes (not applied):**
  1. Have rank 0 broadcast the decoder's decisions (accepted positions, stop flag) each step, so ranks can never disagree.
  2. Or add a cheap per-step consistency check (all-reduce a checksum of the canvas) that raises instead of hanging.
  3. Run the server with `faulthandler` on a signal, so a future hang can dump Python stacks without ptrace.

## 6. Limits
- **Sample size:** 25 tasks × 1 trial. The 16 vs 20 difference is within noise (p = 0.22).
  - More tasks (all 50 retail tasks, airline) and `--num-trials 4` for pass^k would show whether there's a real effect.
- **CACHE run's continuity:** it was restarted twice. Every restart cleared the prefix cache, so the first request after a restart re-encoded its prompt in full. That slightly penalizes CACHE's timing.
- **Per-step timings** are wall-clock on shared hardware, from single runs.
- **The user simulator is an LLM.** Some failures may be the simulator's fault. τ²-bench's own "LLM Judge Review" reported 0 user errors, but it isn't exhaustive.

## 7. Reproduce

Start the server. Add `--polestar --polestar-topk 2 --polestar-refresh-every 8` for the CACHE run:
```bash
cd /scratch/mi72/agentic-dgemma/server && PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp /scratch/mi72/envs/polestar/bin/python serve_openai.py --gpus 0,1 --port 8000 --log-dir ../results/tau2/<label>/server
```
Then run τ²-bench (the OpenAI key is in `/scratch/mi72/.tau2.env`):
```bash
cd /scratch/mi72/tau2-bench && set -a && . /scratch/mi72/.tau2.env && set +a && /scratch/mi72/envs/tau2/bin/tau2 run --domain retail --task-split-name base --num-tasks 25 --num-trials 1 --agent-llm openai/diffusiongemma --agent-llm-args '{"api_base":"http://127.0.0.1:8000/v1","api_key":"x","max_tokens":512}' --user-llm gpt-4.1 --max-concurrency 1 --save-to <label>
```
Summary numbers (writes `summary.json`):
```bash
/scratch/mi72/envs/tau2/bin/python /scratch/mi72/agentic-dgemma/results/tau2/analyze.py
```
