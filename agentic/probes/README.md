# Tool-call probes for DiffusionGemma

`tool_call_probes.jsonl` holds 15 prompts in OpenAI chat format (`messages` + `tools`), each with an
expected outcome (`expect`). They are rendered with the checkpoint's own chat template, which converts
them to Gemma's native tool syntax. Prompts ending in a `tool` message continue the model's turn after
the tool response, so they're rendered without `add_generation_prompt`. All 15 render (85–742 tokens).

| Level | Tests | Probes |
|---|---|---|
| L1 | One call with simple arguments | `L1-weather`, `L1-calc` |
| L2 | Choosing the right tool, and not calling one when none is needed | `L2-select`, `L2-no-call`, `L2-no-call-known` |
| L3 | Typed arguments: arrays, enums, ids | `L3-arrays`, `L3-enum` |
| L4 | Using a tool result: the next call, or an answer | `L4-chain`, `L4-answer-from-result` |
| L5 | τ²-style policy: authenticate first, confirm before writing, refuse out-of-policy requests | `L5-auth-first`, `L5-confirm-before-write`, `L5-refuse-delivered` |
| L6 | SWE-style bash: explore, then fix | `L6-bash-explore`, `L6-bash-edit` |
| L7 | An L1-style call after ~4k tokens of policy text (the runner adds padding from `pad_system_tokens`) | `L7-long-policy` |

`expect` types:
- `call`: the first `<|tool_call>` must name `name`.
  - `arguments` must match exactly, per key.
  - `arguments_contains` maps a key to substrings that must appear in it.
- `no_call`: no tool call; `text_contains` / `text_contains_any` check the reply.
- `no_call_or_call`: either replying or calling a tool in `allowed_calls` is acceptable. `forbidden_calls` must not appear.

Each run records, per probe:
- **format_ok:** the call parses as `call:NAME{...}` and converts to JSON;
- **correct:** the expectation is met;
- **self_response:** the model wrote its own `<|tool_response>`, which the server must cut off;
- **raw output and denoising steps.**

Sweep thinking on/off, `entropy_bound`, `max_denoising_steps`, Polestar-Cache on/off and 3 seeds (see `../PLAN.md`, Phase 0).
