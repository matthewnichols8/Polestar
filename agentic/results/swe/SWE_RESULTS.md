# DiffusionGemma + Polestar on SWE-bench Verified (mini-swe-agent)

*Runs 2026-11-23 → 2026-11-25. Numbers from `analyze_swe.py` → `swe_summary.json`; per-run folders listed at the end.*

## Bottom line

| Configuration | Context cap | **Resolved** | Patches submitted | Submitted → resolved |
|---|---|---|---|---|
| dInfer baseline | 24k | **3/20** (15%) | 5 | 3/5 |
| Polestar CACHE | 24k | **4/20** (20%) | 5 | 4/5 |
| Full Polestar (CACHE + COMMIT) | **18k** | **0/20** | 1 | 0/1 |

- **DiffusionGemma can run a real software-engineering agent loop.** Through an unmodified mini-swe-agent it explores the repository with shell commands, edits files, writes reproduction scripts and submits patches that pass SWE-bench's hidden tests.
- **CACHE vs baseline is a tie** (exact McNemar p = 1.0; the runs differ on only 3 tasks, 1 vs 2). CACHE doesn't hurt task success and is a little faster: about 6% less time per denoising step and 8% less per request (see Speed).
- **Full Polestar can't be compared with the other two.** On 2× 24 GB it runs out of GPU memory past about 20k prompt tokens, so it was rerun with an 18k cap, which cut off 15 of its 20 tasks. Its score measures the memory limit, not COMMIT.
- **Most failures happen before any test runs.** When the agent submits a patch, it usually resolves the task (7 of 10 across the two 24k runs). Most tasks end with no patch, because of malformed tool calls or the context cap.

## Setup

| | |
|---|---|
| Model | `google/diffusiongemma-26B-A4B-it`, int8 experts, expert-parallel on 2× TITAN RTX (24 GB) |
| Server | `agentic-dgemma/server/serve_openai.py`: OpenAI-compatible, native Gemma tool syntax ⇄ OpenAI `tool_calls`, prefix-KV reuse across turns (`--cache-slots 1`), `max_tokens` 1024 (4 canvases of 256) |
| Polestar CACHE | `--polestar --polestar-topk 2 --polestar-refresh-every 8` (8 clusters, delta reuse) |
| Polestar COMMIT | `--polestar-commit`: accept mode; threshold 0.9 / lower bound 0.7; window 3; gate scale 10; drift cap 2; freeze 0.999; `refresh_after_commits` 24 |
| Agent | mini-swe-agent 2.4.6, stock `swebench.yaml` prompts, native `bash` tool; `dgemma_swebench.yaml`: step limit 75, 120 s per command |
| Sandbox | Official `swebench/sweb.eval.*` images, unpacked with `crane` and run under unprivileged `bwrap` (`swe/fakesingularity`); no Docker needed |
| Tasks | 20 SWE-bench Verified instances, seed 42 (`swe/instances.txt`): 10 django, 3 astropy, 2 scikit-learn, 1 each matplotlib, xarray, pylint, pytest, sympy |
| Scoring | `swe/eval_bwrap.py`: the official SWE-bench 5.0.2 patch-apply chain, eval script and grader, run locally. Validated first: reference patches 20/20 resolved, empty patches 0/20 |

## How each task ended

| Exit status | Baseline | CACHE | Full Polestar (18k) |
|---|---|---|---|
| Submitted | 5 | 5 | 1 |
| RepeatedFormatError (malformed tool calls) | 7 | 12 | 4 |
| ContextWindowExceeded (hit the cap) | 7 | 3 | **15** |
| Step limit (75) | 1 | 0 | 0 |
| Median agent steps per task | 27.5 | 26 | 23.5 |
| Requests with a malformed tool call | 3.6% | 5.6% | 5.5% |

**Submitted patches that failed:**
- **django-14034** (baseline and CACHE): the change didn't fix the bug (target test 0/1); existing tests all passed (12/12).
- **sympy-16766** (baseline): the patch didn't apply.
- **django-13401** (full Polestar): the change didn't fix the bug (0/1); existing tests all passed (32/32).

No applied patch broke an existing test.

**Malformed tool calls.** The typical failure is a long multi-line command, such as a heredoc that rewrites a function. The model closes the string, but instead of `<tool_call|>` it continues in prose and reopens the call, so the call can't be parsed. Real excerpt (CACHE run, astropy):

```
    return value_str
EOF
<|"|>}Wait, I should not overwrite the whole file. ... Let me read the whole file.<|"|><|tool_call>call:bash{command:<|"|>cat astropy/io/fits/card.py<|"|>}<tool_call|>
```

The server returns the unparseable call as text. mini-swe-agent replies with a format error, and after repeated errors it ends the task. This happens in only 4–6% of requests, but it ends more tasks than anything else.

**Context cap.** Contexts grow because the agent prints whole files (`cat`) and long test logs. Prefix reuse keeps each turn cheap, but GPU memory still sets a hard limit on total length.

## Speed

Per-request server logs (`server/requests.jsonl`). The ≤18k columns compare the three at the same prompt lengths.

| | Baseline | CACHE | Full Polestar |
|---|---|---|---|
| Requests | 854 | 854 | 713 |
| ms per denoising step (all) | 561 | 522 | — |
| **ms per denoising step (prompt ≤ 18k)** | **530** | **498 (−6.0%)** | **528 (−0.3%)** |
| **s per request (prompt ≤ 18k)** | **10.3** | **9.4 (−8.4%)** | **10.1 (−2.5%)** |
| Denoising steps per request | 20.4 | 18.9 | 19.0 |
| Prompt tokens served from the prefix cache | 91% | 90% | 92% |

**CACHE stats.**
- In the CACHE run, 84% of canvas steps were partial (13,598 partial vs 2,516 full). A partial step recomputes 73% of canvas tokens on average: the tokens that changed plus the 2 clusters with the most drift.
- In full Polestar, only 62% of steps were partial (8,454 vs 5,112 full). 2,978 extra full refreshes were triggered by commits, which cancels most of CACHE's savings. τ²-bench showed the same pattern. `refresh_after_commits` 24 is too eager for 256-token canvases, so tuning it is the obvious next step.

**COMMIT stats.** It accepted 65,129 extra tokens early through drift-gated commits, and froze 245,194 tokens at ≥0.999 confidence.

## Example: a resolved task (CACHE, scikit-learn-26323)

The bug: `ColumnTransformer.set_output` ignores the `remainder` estimator. The agent did the following:
1. Wrote a reproduction script using a heredoc.
2. Grepped for `set_output` in `sklearn/compose/_column_transformer.py`.
3. Paged through the file with `sed -n`.
4. Made two failed multi-line `sed -i` attempts.
5. Wrote a small Python script that edits the file.
6. Re-ran the reproduction script and submitted.

The final patch passes the target test and all 188 existing tests:

```diff
         for trans in transformers:
             _safe_set_output(trans, transform=transform)
 
+        if hasattr(self.remainder, "set_output"):
+            _safe_set_output(self.remainder, transform=transform)
+
         return self
```

## Run history and reliability

- **Fix made before the runs: chat-template ordering.** The template rendered an assistant message's text after its tool responses, which broke prefix reuse. The fix sends the text first, then the calls. It raised reuse from 45% to 92% and cut s/request from 26 to 12 (`server/gemma_format.py`). The τ²-bench runs predate this fix.
- **Baseline:** one out-of-memory crash, at a ~22k prompt. The watchdog restarted the server and the run resumed.
- **CACHE:** the server hung at 01:41 on 11-24 and stayed silent for 17 h. Stack dumps added later showed the cause:
  - If GPU 0 runs out of memory, the main process catches the error and then waits forever to shut down while GPU 1 is still generating, which looks like a hang.
  - If GPU 1 runs out of memory, its process dies, which is a visible crash.
  - The 4 tasks lost to the hang were set aside (`hung-2026-11-24/`) and rerun. One more stall happened later and was restarted automatically.
- **Changes made after the hang:**
  - `run_all.sh` treats 20 minutes of log silence as a hang and restarts the server.
  - The server prints its stack traces on SIGUSR1.
  - An error on the main process now prints the traceback and exits at once.
- **Full Polestar at 24k:** 4 out-of-memory crashes and 2 stalls in 7 tasks, at prompts of 19.5k–21.5k tokens. It was stopped and is kept in `dgemma-polestar-cache-commit-swev20-24k-incomplete/` (1/7 resolved). The rerun at an 18k cap finished without a single crash.

## Limits

- **20 tasks is small.** One task is 5 points, so differences of 1–2 tasks are noise.
- **Scoring is local,** with the official images, eval scripts and grader, but not the official `sb-cli` service. The reference and empty patch checks match the expected results.
- **Context caps:** the baseline and CACHE ran at 24k and full Polestar at 18k, and those caps limit results directly. On TITAN RTX hardware this is about the memory ceiling, and all timings are TITAN RTX timings.
- **No thinking mode;** greedy agent settings; one attempt per task.

## Next steps

1. **Make long tool calls more reliable:** recover calls with a missing `<tool_call|>` on the server, or steer the model toward shorter commands. This is the single biggest lever.
2. **Run all three configurations at 24k or more on the H200 (96 GB),** for a fair full-Polestar comparison.
3. **Tune `refresh_after_commits`,** so that COMMIT stops cancelling CACHE's savings.
4. **Trim old tool outputs** from the history to slow context growth.

## Files

`results/swe/`:
- `dgemma-dinfer-swev20/`
- `dgemma-polestar-cache-swev20/`
- `dgemma-polestar-cache-commit-swev20/`
- `dgemma-polestar-cache-commit-swev20-24k-incomplete/`
- `check-gold/`, `check-empty/`
- `run_all.log`
- `analyze_swe.py`, `swe_summary.json`

Each run folder has:
- `config.json`, `preds.json`
- the trajectories, `<task>/<task>.traj.json`
- `eval/summary.json` and the per-task `eval/<task>/` (patch, test output, report)
- `server/requests.jsonl`
- `server-N.log`

Harness code is in `agentic-dgemma/swe/`; server code in `agentic-dgemma/server/`.
