# Phase 0 results: DiffusionGemma tool-call probes

**Setup:**
- 15 probes (`probes/tool_call_probes.jsonl`, levels L1–L7);
- 5 decoding configs × 3 seeds = 225 generations;
- `google/diffusiongemma-26B-A4B-it`, int8 experts on 2× TITAN RTX;
- dInfer `CanvasDiffusionLLM` from the Polestar fork (`feature/diffusion-gemma`);
- runner: `probes/run_probes.py`; raw outputs in `phase0/raw.jsonl`; rescored in `phase0/raw_rescored.jsonl`.

## Summary (rescored; see "Scoring changes" below)

| config | correct | L1 | L2 | L3 | L4 | L5 | L6 | L7 | format ok (calls) | self tool_response | s/probe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 98% | 100% | 100% | 83% | 100% | 100% | 100% | 100% | 100% | 60% | 2.7 |
| thinking | 91% | 100% | 100% | 50% | 83% | 100% | 100% | 100% | 100% | 60% | 4.4 |
| eb0.05 | 98% | 100% | 100% | 83% | 100% | 100% | 100% | 100% | 100% | 60% | 2.9 |
| steps96 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 60% | 2.8 |
| polestar | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 60% | 2.6 |

- **base:** `entropy_bound` 0.1, 48 max steps, 512 tokens, no thinking, no cache.
- **thinking:** `enable_thinking=True`, 1024 tokens.
- **eb0.05:** `entropy_bound` 0.05.
- **steps96:** 96 max denoising steps.
- **polestar:** Polestar-Cache, `topk=2`, `refresh_every=8`.

Each cell is 6 generations (2 probes × 3 seeds) for L1, L3, L4 and L6; 9 for L2 and L5; 3 for L7.

## Findings
1. **Tool-call syntax was never malformed:** 0 of 225 generations. Every `<|tool_call>call:NAME{...}<tool_call|>` parsed and converted to JSON arguments, including arrays, enums, ids and a bash command containing quotes.
2. **Tool choice, abstention and policy behaviour are good:**
   - It never calls a tool when none is needed (L2).
   - It authenticates first and asks for confirmation before write actions (L5).
   - It refuses to cancel a delivered order (L5).
   - It finds the bug and edits the right line with `sed` (L6).
   - It still makes the right call after ~4k tokens of policy text (L7).
3. **Real errors (3 of 225):**
   - **A dropped character in a long argument:** in 2 runs, `order_id` came out as `W4597054` instead of `#W4597054` in the 4-argument exchange call (base seed 2, eb0.05 seed 0). It's the kind of small token loss parallel decoding can cause, and the most important failure mode to track on τ²-bench, where ids must match exactly.
   - **Thinking with no answer:** once (thinking, L4-chain) the model produced only thinking and no call or reply.
4. **Thinking mode doesn't help here:**
   - About 2× the denoising steps (9.8 vs 4.9 per probe) and 1.6× the time.
   - Lower accuracy. In 3 of its 6 L3 failures it first looked up the order before the exchange, a defensible extra step that the probe didn't expect.
5. **The model writes its own `<|tool_response>` after 60% of calls.** The decoder already stops there (token 50 is an end token), so the output ends cleanly at the call. The Phase 1 server must keep this cut-off.
6. **Decoding settings barely matter at this scale.** steps96 and polestar scored 100%; base and eb0.05 98% (one `#` miss each). That's within noise for 45 generations per config.
   - Polestar-Cache didn't hurt and was the fastest at 2.6 s per probe.
   - Tool calls finish in 2–4 denoising steps (~1–2 s), because one call fits in a single canvas.

## Scoring changes and caveats
- **The first quick run** (base, one seed) failed 4 probes because of **probe flaws**, which were fixed before this sweep:
  - L3-arrays and L3-enum: the prompt claimed authentication, but the policy requires email authentication, so the model correctly asked for the email.
  - L4-chain: "most recent order" was ambiguous with an undated list.
  - L6-bash-edit: the prompt didn't say the edit was the next step.
- **After the sweep,** L5-confirm-before-write's text check was widened. The model asked "Would you like me to cancel this order…?" (a correct confirmation request), but the check only looked for "confirm", "yes" or "proceed". The original scores are kept in `phase0/summary_original_scorer.md` (base 96%, steps96 96%, polestar 100%).
- **These are small, hand-written probes** and the model passed them after the fixes, so they are a **smoke test, not a benchmark.**
  - Each probe is a single model turn at a fixed point in a conversation.
  - The real test is closed-loop multi-turn dialogue with a simulated user (τ²-bench) and long, growing contexts (SWE-bench).

## Decision
**The Phase 0 gate passed:** L1–L3 format validity was 100% (≥ 90% required) and L5 success was non-zero. Proceed to Phase 1 (the OpenAI-compatible server with prefix reuse). Structured-output safeguards (Phase 1c) aren't needed for syntax. Track exact-id fidelity, e.g. the dropped `#`, on τ²-bench.
