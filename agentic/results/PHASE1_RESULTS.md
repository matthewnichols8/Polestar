# Phase 1 results: OpenAI-compatible server with prefix reuse

**Setup:**
- `google/diffusiongemma-26B-A4B-it`, int8 experts on 2× TITAN RTX;
- dInfer `CanvasDiffusionLLM` with entropy bound 0.1, 48 max steps, no thinking, no Polestar-Cache;
- server: `server/serve_openai.py`; checks: `server/phase1_check.py`, `server/litellm_check.py`;
- raw outputs in `phase1/` (`probes.jsonl`, `multiturn.json`, `summary.json`, `server/requests.jsonl`).

## What was built
1. **Prefix reuse in dInfer** (Polestar fork, branch `feature/dgemma-prefix-reuse`, commit `9af281c`, not pushed):
   - `CanvasDiffusionLLM.generate(..., prefix_cache=...)` encodes only the prompt tokens after the cached prefix.
   - `last_prompt_cache` is a snapshot taken right after the prompt is encoded.
   - `PerLayerKVCache.snapshot()` shares tensors, so it costs no memory until the copies diverge.
   - **CPU tests:**
     - reuse is **bitwise identical** to encoding the prompt in two pieces;
     - it matches a one-pass encode up to float rounding;
     - the caller's snapshot is never modified.
   - Full suite: 93 passed, 8 skipped.
2. **`server/gemma_format.py`:** OpenAI ↔ Gemma translation.
   - Requests are rendered with the checkpoint's chat template. JSON-string `arguments` are parsed, and text-only content parts are flattened.
   - Output is parsed into OpenAI `tool_calls` (`arguments` as a JSON string) plus `reasoning_content`.
   - Malformed calls are returned as text.
   - Only real special tokens are stripped, so text like `<module>` and `<div>` is kept.
   - 10 CPU tests use the real tokenizer, including: **every turn's prompt extends the previous one**, with tools (τ²-style) and without (mini-swe-agent style).
3. **`server/serve_openai.py`:** FastAPI `/v1/chat/completions` and `/v1/models`.
   - **GPUs and requests:**
     - Expert parallel across both GPUs: rank 0 serves HTTP and broadcasts each request.
     - One request at a time.
     - Followers exit if the launcher dies.
   - **Request options:**
     - `max_tokens` is rounded up to whole canvases.
     - `seed` re-seeds the decoder.
     - `stop` is honoured; `tool_choice: "none"` is supported.
     - `chat_template_kwargs.enable_thinking` turns thinking on.
     - Streaming is supported as a single chunk.
     - The non-standard `cache_reuse: false` forces a full encode.
   - **Prompt caches:** the last 4 are kept (`--cache-slots`); the longest one the new prompt starts with is reused.
   - **Diagnostics** go in a `dgemma` response field and `requests.jsonl`.

## Results
**A. Phase 0 probes through the HTTP API** (15 probes × 3 seeds):
- **45/45 correct** (Phase 0's same config, called directly: 98%).
- `finish_reason`: 27 `tool_calls`, 18 `stop`.
- 2.6 s per probe.
- No self-written tool response reached a client: the decoder stops at `<|tool_response>`, and the parser would drop it anyway.

**B. A growing SWE-style conversation** (9 turns; each turn adds a `cat` of a ~1.5–2k-token source file; the same prompts in both modes):

| turn | prompt tokens | new tokens encoded (reuse) | s/turn reuse | s/turn no reuse | s/step reuse | s/step no reuse |
|---|---|---|---|---|---|---|
| 0 | 63 | 63 | 3.2 | 2.6 | 0.36 | 0.37 |
| 2 | 1,318 | 981 | 5.5 | 6.1 | 0.46 | 0.51 |
| 4 | 4,862 | 1,828 | 7.7 | 12.3 | 0.59 | 0.82 |
| 6 | 8,621 | 1,911 | 10.9 | 18.4 | 0.60 | 1.23 |
| 8 | 11,927 | 2,001 | **11.8** | **26.1** | 0.66 | 1.45 |
| **total (9 turns)** | | | **64.4** | **111.6** | | |

- **Reuse matched every turn:** the whole previous prompt was reused (`reused = previous prompt length`).
- **Encoding costs ~1.2 ms per token** on the TITANs. Without reuse, re-encoding 12k tokens adds ~15 s per turn and the cost grows with the conversation. With reuse, per-turn time depends on the *new* tokens plus denoising.
- **Denoising itself slows with context:** ~0.36 s/step at 63 tokens vs ~0.55 s/step at 12k (attention over the cache). That cost remains, and it's where Polestar-Cache / the canvas cache could help on long contexts (Phase 3).
- Step counts differ slightly between the two modes. Reuse and a one-pass encode differ by float rounding, which can change the sampled canvas. Hence the per-step columns.

**C. API behaviour:**
- streaming works ("Hello to you.", `finish_reason: stop`);
- `tool_choice: "none"` produced no calls;
- `n=2` and malformed `arguments` get HTTP 400.

**D. LiteLLM** (`openai/diffusiongemma`; the client path used by τ²-bench and mini-swe-agent):
- **Turn 1:** a `get_order_details` call with `{"order_id": "#W4597054"}`, finish `tool_calls`.
- **Turn 2:** the assistant message LiteLLM returned was appended as-is (`model_dump`) plus a tool result. The model answered "Your order #W4597054 (Desk Lamp) has been delivered." and **reused all 124 tokens** of turn 1's prompt.

## Caveats
- **Reuse covers the previous prompt only, not the model's own previous output.** That output reappears re-rendered in the next prompt; for example, the template sorts tool-call argument keys. Those tokens (≤ one canvas per turn) are re-encoded, which is cheap.
- **One request at a time**, batch 1. Agent runs must use one worker (τ² `--max-concurrency 1`, mini-swe-agent `--workers 1`); concurrent requests queue.
- **`temperature`/`top_p` are ignored:** the decoder has its own noise schedule.
- **Prefix reuse when thinking is on is untested.** The template drops earlier thinking from past turns, which may break the prefix match; the `common_prefix_tokens` diagnostic will show it.
- **New environment `/scratch/mi72/envs/agents`** (Python 3.11, `litellm`): the current LiteLLM release needs `mcp` installed separately.

## Decision
**Phase 1 is done.** The server is usable by LiteLLM-based harnesses, and agent conversations reuse their prefix each turn (2.2× faster at 12k tokens; the gap widens with length). Next: Phase 2 (τ²-bench), which needs the user-simulator API key, and Phase 3 (mini-swe-agent / SWE-bench), which needs Docker or apptainer.
