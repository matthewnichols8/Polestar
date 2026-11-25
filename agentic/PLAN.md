# DiffusionGemma on agentic tasks (τ²-bench, SWE-bench): research plan

## Goal
Measure whether `google/diffusiongemma-26B-A4B-it`, running in our dInfer integration (`/scratch/mi72/Polestar`, branch `feature/diffusion-gemma`), can do multi-turn tool use.
- **Benchmarks:** τ²-bench (customer service with policies and tools) and SWE-bench (fixing real GitHub issues from a bash shell).
- **Secondary questions:**
  - How does parallel diffusion decoding behave on *structured* output (tool calls, JSON, shell commands)?
  - Do Polestar-Cache and prefix reuse help long, growing agent contexts?

## What we know already
- **The model supports tool calling natively.** Its chat template declares tools in the system turn and emits calls in Gemma's own syntax, not JSON. Rendered by the checkpoint's template:
  ```
  <|turn>system ... <|tool>declaration:get_order{description:<|"|>...<|"|>,parameters:{...}}<tool|><turn|>
  <|turn>model
  <|tool_call>call:get_order{order_id:<|"|>#W123<|"|>}<tool_call|><|tool_response>response:get_order{value:<|"|>...<|"|>}<tool_response|>
  ```
  - The tool result is placed *inside the model turn*, right after the call, and the model continues the same turn.
  - The template also has a thinking mode (`<|think|>`).
- **Generation:** 256-token canvases, denoised in parallel. A tool call is decoded all at once with the surrounding text, not left to right.
- **This server:** 2× TITAN RTX with int8 experts runs at ~30–35 tok/s at 512 tokens, ~0.5 s per denoising step.
  - Prompts are re-encoded on every request; there's no cross-request prefix reuse yet.
  - **No Docker access** for `mi72`: permission denied on `/var/run/docker.sock`; no podman or apptainer installed.
- **τ²-bench** (sierra-research/tau2-bench):
  - Models are called through LiteLLM (`--agent-llm`, `--agent-llm-args`), so an OpenAI-compatible local server works.
  - It also needs a **user-simulator LLM** (`--user-llm`, default GPT-class) to play the customer.
  - Domains: `mock`, `airline`, `retail`, `telecom` (+ newer ones); 50 frozen tasks per domain in text-v1.
- **SWE-bench via `mini-swe-agent`:** bash-only, so no tool-calling API is needed. It uses LiteLLM and runs commands in docker/podman/apptainer/bubblewrap. Evaluating the patches needs the SWE-bench Docker harness, or the hosted `sb-cli` evaluation.

## Phase 0: Tool-call probes (offline, no benchmark infrastructure). ✅ Done; see `results/PHASE0_RESULTS.md`
**Result:** 216–219 of 225 correct depending on scoring, and 0 malformed tool calls. Thinking mode is slower and slightly worse. The main failure mode is a dropped character in a long id (`#W…` → `W…`). The gate passed: proceed to Phase 1 without structured-output safeguards.

Run `probes/tool_call_probes.jsonl` (levels L1–L7, described in `probes/README.md`) straight through `CanvasDiffusionLLM`, rendering with the model's own chat template.
1. **Script `probes/run_probes.py`:** render each probe, generate, parse the first `<|tool_call>…<tool_call|>`, and score:
   - format validity: parses as Gemma tool syntax, and converts to JSON arguments;
   - tool choice;
   - arguments;
   - "should not call" cases;
   - multi-call ordering;
   - policy adherence (asks for confirmation before a write action).
2. **Settings to sweep:**
   - thinking on/off;
   - `entropy_bound` 0.1 vs 0.05;
   - `max_denoising_steps` 48 vs 96;
   - Polestar-Cache off/on;
   - 3 seeds.
3. **Failure taxonomy:**
   - malformed syntax (unbalanced `{`, missing `<|"|>`);
   - hallucinated tool names or arguments;
   - calls the model *answers itself* (it writes its own `<|tool_response>`);
   - parallel-decoding artifacts (duplicated or interleaved tokens, the "she she" effect we saw in GSM8K).
4. **Decision gate:** if L1–L3 format validity is < 90%, fix decoding (Phase 1c) before any benchmark. If L5 (multi-turn) success is near 0, τ²-bench results will be ~0; report that as the finding instead of spending API budget.

## Phase 1: Serving layer. ✅ Done; see `results/PHASE1_RESULTS.md`
**Result:**
- The server is in `server/`. Probes through HTTP scored 45/45, and the LiteLLM tool round trip works.
- Prefix reuse matched every turn: a 12k-token turn takes 11.8 s vs 26.1 s without it (64 vs 112 s over 9 turns).
- 1c (safeguards) was skipped: Phase 0 had no format failures.

**1a. OpenAI-compatible server** (`serve_openai.py`): FastAPI `/v1/chat/completions` over the expert-parallel dInfer model.
- Rank 0 serves HTTP and broadcasts prompts; the other ranks follow. This is the same pattern as the GSM8K benchmark.
- **Request handling:**
  - Render `messages` + `tools` with the chat template, converting OpenAI `tool_calls` and `tool` messages to Gemma's inline format.
  - Generate.
  - **Stop at the first `<tool_call|>`**: discard anything after it, especially a self-written `<|tool_response>`.
  - Parse Gemma call syntax into OpenAI `tool_calls`, with `arguments` as a JSON string.
  - Return `finish_reason="tool_calls"`.
- Honour `max_tokens` (rounded up to whole canvases) and `temperature` (ignored; the sampler has its own schedule; document this).

**1b. Prefix reuse across turns.** Agent conversations grow by appending, and our encode is causal and appendable.
- Keep the last request's token ids and per-layer KV cache.
- If the new prompt starts with the old one, encode only the new suffix.
- This turns per-turn prefill from O(conversation) into O(new tokens), which is essential for SWE-bench contexts of 10–30k tokens.
- Verify it bit-for-bit against a full encode on CPU with the tiny-model tests in the Polestar repo.

**1c. Structured-output safeguards** (only if Phase 0 shows format failures):
- **Validate and retry:** if the parse fails, regenerate with a different seed, up to N tries.
- **More careful decoding on tool turns:** a lower `entropy_bound`, i.e. fewer tokens accepted per step.
- **Syntax repair:** close braces and quote delimiters.
- **Research option:** commit the tool-call skeleton (`<|tool_call>call:NAME{`) first, then denoise the arguments. Diffusion makes classic left-to-right grammar-constrained decoding awkward; that's an open question worth reporting.

## Phase 2: τ²-bench
1. **Install** `tau2-bench` in its own env (`/scratch/mi72/envs/tau2`).
2. **Point the agent model at the server:**
   ```
   --agent-llm openai/diffusiongemma
   --agent-llm-args '{"api_base":"http://localhost:8000/v1","api_key":"x"}'
   ```
3. **User simulator:** an API model, e.g. GPT-4.1 or Claude. That needs an API key from you and costs money; estimate it from the token counts in the `mock` run.
4. **Order:**
   - `mock` domain, `--num-tasks 5`: plumbing check;
   - `retail` and `airline`, 10 tasks, 1 trial;
   - the full 50 tasks × `--num-trials 4` (pass^k) only if the 10-task scores are non-trivial.
5. **Compare:** dInfer vs dInfer + Polestar-Cache, and thinking on/off. Report pass^1, the tool-call parse-failure rate, turns per task, and wall time.
6. **Time estimate:** ~10–20 agent turns per task × ~10–20 s per turn on the TITANs, so ~3–5 min per task; 50 tasks ≈ 3–4 h per domain per trial.

## Phase 3: SWE-bench (Lite / Verified subset)
**Blocker:** no Docker access. Ask the server admin to add `mi72` to the `docker` group, or install apptainer. Alternatives:
- run the agent's environment with `bubblewrap` (supported by mini-swe-agent) and score with hosted `sb-cli`;
- run on a machine where Docker is available.

1. **Agent:** `mini-swe-agent` (bash-only) with the LiteLLM model pointed at our server. Bash-only avoids the tool-call format entirely. If needed, adjust its prompt template so the model's ```` ```bash ```` block style matches what the model produces.
2. **Start small:**
   - 10 SWE-bench Verified instances with step and cost limits (e.g. 50 steps);
   - check trajectories for loops, malformed commands and context growth;
   - then 50 instances.
3. **Context:** contexts reach 10–30k tokens. Prefix reuse (1b) is required. Measure prefill vs denoise time per step, and test Polestar-Cache on these long prefixes.
4. **Time estimate:** ~30–50 steps per instance × 10–30 s per step ≈ 10–25 min per instance, so 50 instances ≈ 10–20 h.

## Phase 4: Write-up
- **Tool-call reliability under parallel decoding:** probe results and failure taxonomy. This connects to ParallelBench's finding that parallel decoding hurts tightly constrained outputs.
- **Agentic scores:** τ²-bench pass^1 and SWE-bench resolve rate vs published autoregressive models of similar active size (~4B active), with the hardware stated.
- **System measurements:** prefix-reuse savings and the Polestar-Cache effect on long agent contexts.

## Needed from you
1. **API key and budget** for the τ²-bench user simulator.
2. **Docker group or apptainer** from the server admin, for SWE-bench.
3. **Where this lives:** keep it in `/scratch/mi72/agentic-dgemma`, or put it in a repo (e.g. a `research/agentic` branch of your Polestar fork).
