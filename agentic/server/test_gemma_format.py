"""CPU tests for the OpenAI <-> Gemma translation, using the real DiffusionGemma tokenizer and chat template.

    PYTHONPATH= HF_HOME=/scratch/mi72/hf /scratch/mi72/envs/polestar/bin/python -m pytest -q server
"""

import json

import pytest

from gemma_format import (RequestError, apply_stop, gemma_special_tokens, openai_message, parse_output, render,
                          to_template_messages)

TOOLS = [{"type": "function", "function": {
    "name": "get_order_details", "description": "Look up an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}}]


@pytest.fixture(scope="module")
def tokenizer():
    from huggingface_hub import snapshot_download
    from dinfer.model import load_diffusion_gemma_tokenizer

    return load_diffusion_gemma_tokenizer(snapshot_download("google/diffusiongemma-26B-A4B-it"))


def _call(arguments, call_id="call_1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "get_order_details", "arguments": arguments}}]}


def _decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False)


# ---------------------------------------------------------------------------------------------- parsing
def test_parse_tool_call_and_drop_self_written_response():
    text = ('Let me check.<|tool_call>call:get_order_details{order_id:<|"|>#W123<|"|>}<tool_call|>'
            '<|tool_response>response:get_order_details{value:<|"|>made up<|"|>}')
    parsed = parse_output(text)
    assert parsed["calls"] == [{"name": "get_order_details", "arguments": {"order_id": "#W123"}}]
    assert parsed["content"] == "Let me check."
    assert parsed["self_tool_response"] and "made up" not in json.dumps(parsed)
    message = openai_message(parsed)
    assert json.loads(message["tool_calls"][0]["function"]["arguments"]) == {"order_id": "#W123"}


def test_parse_typed_arguments_and_thinking():
    text = ('<|channel>thought\nneed both<channel|><|tool_call>call:f{ids:[<|"|>a<|"|>,<|"|>b<|"|>],n:3,'
            'ok:true,cmd:<|"|>grep -n "x: y" a.py<|"|>}<tool_call|>')
    parsed = parse_output(text)
    assert parsed["thinking"] == "need both"
    assert parsed["calls"][0]["arguments"] == {"ids": ["a", "b"], "n": 3, "ok": True, "cmd": 'grep -n "x: y" a.py'}
    assert openai_message(parsed)["reasoning_content"] == "need both"


def test_malformed_call_is_returned_as_text():
    parsed = parse_output('<|tool_call>call:f{x:<|"|>unterminated}')
    assert not parsed["calls"] and parsed["call_errors"]
    message = openai_message(parsed)
    assert "tool_calls" not in message and "call:f" in message["content"]


def test_plain_reply_and_stop():
    parsed = parse_output("THOUGHT: look\n```bash\nls\n```<turn|>")
    assert parsed["content"] == "THOUGHT: look\n```bash\nls\n```" and not parsed["calls"]
    assert apply_stop("abc STOP def", ["STOP"]) == ("abc ", True)
    assert apply_stop("abc", None) == ("abc", False)


def test_special_tokens_are_not_ordinary_text(tokenizer):
    specials = gemma_special_tokens(tokenizer)
    assert "<|tool_call>" in specials and "<turn|>" in specials
    # Stripping must never eat code/HTML/traceback text such as "<module>" or "<div>".
    content = parse_output("File x, line 1, in <module>\n<div>hi</div><b>x</b>", specials)["content"]
    assert content == "File x, line 1, in <module>\n<div>hi</div><b>x</b>"


# ---------------------------------------------------------------------------------------------- requests
def test_string_arguments_are_parsed_for_the_template():
    messages = to_template_messages([_call('{"order_id": "#W1"}')])
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {"order_id": "#W1"}
    with pytest.raises(RequestError):
        to_template_messages([_call("{not json")])
    with pytest.raises(RequestError):
        to_template_messages([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}])


def test_render_matches_native_syntax(tokenizer):
    ids = render(tokenizer, [{"role": "user", "content": "Order #W1?"}, _call('{"order_id": "#W1"}'),
                             {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "delivered"}'}],
                 TOOLS)
    text = _decode(tokenizer, ids)
    assert '<|tool_call>call:get_order_details{order_id:<|"|>#W1<|"|>}<tool_call|>' in text
    assert "<|tool_response>response:get_order_details" in text
    assert text.endswith("<tool_response|>")  # the model continues its own turn after a tool response


def _conversation():
    """A tau2-style exchange, one prompt per agent turn, as an OpenAI client would send it."""
    system = {"role": "system", "content": "You are a retail agent. Authenticate users first."}
    turns = [
        [{"role": "user", "content": "Hi, where is order #W1?"}],
        [_call('{"order_id": "#W1"}'), {"role": "tool", "tool_call_id": "call_1", "content": '{"status": "x"}'}],
        [{"role": "assistant", "content": "It shipped yesterday."}, {"role": "user", "content": "Thanks!"}],
        [_call('{"order_id": "#W2"}', "call_2"),
         {"role": "tool", "tool_call_id": "call_2", "content": '{"status": "y"}'}],
    ]
    messages, prompts = [system], []
    for turn in turns:
        messages = messages + turn
        prompts.append(list(messages))
    return prompts


@pytest.mark.parametrize("with_tools", [True, False])
def test_each_turn_extends_the_previous_prompt(tokenizer, with_tools):
    """Prefix reuse depends on this: the prompt of turn k+1 starts with the prompt of turn k."""
    rendered = [render(tokenizer, p, TOOLS if with_tools else None) for p in _conversation()]
    for previous, current in zip(rendered, rendered[1:]):
        assert current[:len(previous)] == previous, (
            f"diverges at token {next(i for i, (a, b) in enumerate(zip(previous, current)) if a != b)}")


def test_bash_agent_turns_extend_the_previous_prompt(tokenizer):
    """mini-swe-agent style: no tools, plain alternating user/assistant text."""
    messages = [{"role": "system", "content": "You are a helpful assistant that can run bash."},
                {"role": "user", "content": "Fix the bug in calc.py."}]
    previous = render(tokenizer, messages)
    for step in range(3):
        messages = messages + [{"role": "assistant", "content": f"THOUGHT: step {step}\n```bash\nls\n```"},
                               {"role": "user", "content": f"<returncode>0</returncode>\n<output>a.py</output>"}]
        current = render(tokenizer, messages)
        assert current[:len(previous)] == previous
        previous = current


BASH = [{"type": "function", "function": {
    "name": "bash", "description": "Run a bash command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def _bash(call_id, command):
    return {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": command})}}


def test_text_plus_tool_call_turns_extend_the_previous_prompt(tokenizer):
    """mini-swe-agent v2 style: every reply is reasoning text plus native `bash` tool calls. The template puts an
    assistant message's text after its tool responses, which broke prefix reuse; the text must render before the
    calls (the order the model generated them) and each prompt must extend the previous one."""
    messages = [{"role": "system", "content": "You can run bash."}, {"role": "user", "content": "Fix the bug."}]
    steps = [
        [{"role": "assistant", "content": "Let me look around.", "tool_calls": [_bash("c1", "ls")]},
         {"role": "tool", "tool_call_id": "c1", "content": "a.py\nb.py"}],
        [{"role": "assistant", "content": "Two reads at once.", "tool_calls": [_bash("c2", "cat a.py"),
                                                                               _bash("c3", "cat b.py")]},
         {"role": "tool", "tool_call_id": "c2", "content": "print(1)"},
         {"role": "tool", "tool_call_id": "c3", "content": "print(2)"}],
        [{"role": "assistant", "content": None, "tool_calls": [_bash("c4", "grep -n x a.py")]},
         {"role": "tool", "tool_call_id": "c4", "content": ""}],
        [{"role": "assistant", "content": "I think I am done."},
         {"role": "user", "content": "Tool call error: every response needs a bash tool call."}],
        [{"role": "assistant", "content": "Submitting.", "tool_calls": [_bash("c5", "echo DONE")]},
         {"role": "tool", "tool_call_id": "c5", "content": "DONE"}],
    ]
    previous = render(tokenizer, messages, BASH)
    for step in steps:
        messages = messages + step
        current = render(tokenizer, messages, BASH)
        assert current[:len(previous)] == previous, (
            f"diverges at token {next(i for i, (a, b) in enumerate(zip(previous, current)) if a != b)}: "
            f"{_decode(tokenizer, previous[-20:])!r}")
        previous = current
    text = _decode(tokenizer, previous)
    assert text.index("Let me look around.") < text.index("call:bash{command:<|\"|>ls<|\"|>}")
    assert text.index("Two reads at once.") < text.index("cat a.py") < text.index("cat b.py")
    assert text.index("Submitting.") < text.index("echo DONE")
