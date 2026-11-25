"""Translate between OpenAI chat-completions requests/responses and DiffusionGemma's native chat format.

Requests: OpenAI `messages` + `tools` are rendered with the checkpoint's own chat template, which already
understands OpenAI-style `tool_calls` / `role: tool` messages; it only needs `function.arguments` as a mapping,
not the JSON string OpenAI clients send.

Responses: the model writes tool calls as `<|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>` and
thinking as `<|channel>thought ...<channel|>`. Anything after the first self-written `<|tool_response>` is the
model imagining the tool's answer and is dropped (the decoder already stops there: token 50 is an end token).
"""

from __future__ import annotations

import copy
import json
import re
import uuid

STR = '<|"|>'
TOOL_RESPONSE = "<|tool_response>"
TURN_END = "<turn|>"


class RequestError(ValueError):
    """The request cannot be rendered (HTTP 400)."""


# ---------------------------------------------------------------------------------------------- requests
def _flatten_content(content):
    """OpenAI content may be a list of parts; the server is text-only."""
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif part.get("type") in ("text", "input_text", "output_text"):
                parts.append(part.get("text", ""))
            else:
                raise RequestError(f"unsupported content part type {part.get('type')!r} (text only)")
        return "".join(parts)
    raise RequestError(f"unsupported content {type(content).__name__}")


def to_template_messages(messages):
    """OpenAI messages -> messages the DiffusionGemma chat template accepts."""
    out = []
    for message in messages:
        m = copy.deepcopy(message)
        role = m.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise RequestError(f"unsupported role {role!r}")
        if role == "developer":
            m["role"] = "system"
        m["content"] = _flatten_content(m.get("content"))
        for call in m.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    function["arguments"] = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError as exc:
                    raise RequestError(f"tool call arguments are not JSON: {arguments[:200]!r}") from exc
        if not m.get("tool_calls"):
            m.pop("tool_calls", None)
        if m["role"] == "assistant" and m.get("tool_calls") and m["content"].strip():
            # The template renders an assistant message's text *after* its tool responses, and closes the turn
            # there only while nothing follows - so (a) the model sees its reasoning out of order and (b) the
            # previous prompt stops being a prefix of the next one, which defeats prefix reuse. Two consecutive
            # assistant messages render as one model turn (continue_same_model_turn), so send the text first and
            # the calls second: `text<|tool_call>...<tool_call|><|tool_response>...`, the order the model wrote.
            out.append({"role": "assistant", "content": m["content"]})
            m = {**m, "content": ""}
        out.append(m)
    return out


def render(tokenizer, messages, tools=None, enable_thinking=False):
    """Token ids for the prompt. `add_generation_prompt=True` is right in every case: after a tool response the
    template adds nothing (the model continues its own turn), otherwise it opens a model turn."""
    try:
        return tokenizer.apply_chat_template(to_template_messages(messages), tools=tools or None,
                                             add_generation_prompt=True, tokenize=True,
                                             enable_thinking=enable_thinking)
    except RequestError:
        raise
    except Exception as exc:  # jinja raise_exception, malformed messages
        raise RequestError(f"chat template failed: {exc}") from exc


# ---------------------------------------------------------------------------------------------- responses
def gemma_args_to_python(body: str):
    """Parse Gemma's argument syntax `{key:<|"|>text<|"|>,n:3,xs:[<|"|>a<|"|>]}` into Python."""
    strings = []

    def stash(match):
        strings.append(match.group(1))
        return f"\x00{len(strings) - 1}\x00"

    text = re.sub(re.escape(STR) + r"(.*?)" + re.escape(STR), stash, body, flags=re.S)
    if STR in text:
        raise ValueError("unbalanced string delimiter")
    text = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:", r'\1"\2":', text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: json.dumps(strings[int(m.group(1))]), text)
    return json.loads(text)


def parse_output(text: str, special_tokens=()):
    """Split decoded model output (special tokens kept) into thinking, visible content and tool calls.

    Returns dict(thinking, content, calls=[{name, arguments}], call_errors=[raw...], self_tool_response)."""
    self_response = TOOL_RESPONSE in text
    text = text.split(TOOL_RESPONSE, 1)[0].split(TURN_END, 1)[0]
    thinking = "".join(m.group(1) for m in re.finditer(r"<\|channel>(.*?)(?:<channel\|>|$)", text, flags=re.S))
    thinking = re.sub(r"^thought\n", "", thinking)
    visible = re.sub(r"<\|channel>.*?(?:<channel\|>|$)", "", text, flags=re.S)
    calls, errors = [], []
    for match in re.finditer(r"<\|tool_call>(.*?)(<tool_call\|>|$)", visible, flags=re.S):
        raw, closed = match.group(1), bool(match.group(2))
        m = re.match(r"\s*call:([A-Za-z_][A-Za-z0-9_.\-]*)\s*(\{.*\})\s*$", raw, flags=re.S)
        if not closed or not m:
            errors.append(raw)
            continue
        try:
            calls.append({"name": m.group(1), "arguments": gemma_args_to_python(m.group(2))})
        except (ValueError, json.JSONDecodeError):
            errors.append(raw)
    content = re.sub(r"<\|tool_call>.*?(?:<tool_call\|>|$)", "", visible, flags=re.S)
    for token in special_tokens:
        content = content.replace(token, "")
    return {"thinking": thinking.strip(), "content": content.strip(), "calls": calls, "call_errors": errors,
            "self_tool_response": self_response}


def apply_stop(content: str, stop):
    """OpenAI `stop`: cut the content at the first stop string. Returns (content, hit)."""
    if not stop:
        return content, False
    stops = [stop] if isinstance(stop, str) else list(stop)
    cut = min((content.find(s) for s in stops if s and s in content), default=-1)
    return (content[:cut], True) if cut >= 0 else (content, False)


def openai_message(parsed, malformed_as_text=True):
    """Build the OpenAI assistant message and finish_reason hint from `parse_output`'s result."""
    message = {"role": "assistant", "content": parsed["content"] or None}
    if parsed["thinking"]:
        message["reasoning_content"] = parsed["thinking"]
    if parsed["calls"]:
        message["tool_calls"] = [
            {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
            for c in parsed["calls"]
        ]
    elif parsed["call_errors"] and malformed_as_text:
        # Surface an unparseable call as text so the agent (and its logs) can see what happened.
        raw = "\n".join(f"<|tool_call>{e}" for e in parsed["call_errors"])
        message["content"] = ((message["content"] or "") + "\n" + raw).strip()
    return message


def gemma_special_tokens(tokenizer):
    """Special-token strings to strip from visible content (longest first)."""
    tokens = {t.content for t in tokenizer.added_tokens_decoder.values() if t.special}
    tokens |= set(getattr(tokenizer, "all_special_tokens", []))
    return sorted((t for t in tokens if t), key=len, reverse=True)
