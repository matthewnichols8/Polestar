"""Drive a two-turn tool-calling exchange through LiteLLM (the client path tau2-bench and mini-swe-agent use).

    PYTHONPATH= /scratch/mi72/envs/agents/bin/python litellm_check.py
"""

import json

import litellm

KW = dict(model="openai/diffusiongemma", api_base="http://127.0.0.1:8000/v1", api_key="x", max_tokens=256, seed=0)
TOOLS = [{"type": "function", "function": {
    "name": "get_order_details", "description": "Get the status and items of an order.",
    "parameters": {"type": "object", "properties": {"order_id": {"type": "string", "description": "e.g. #W1234567"}},
                   "required": ["order_id"]}}}]

messages = [
    {"role": "system", "content": "You are a retail support agent. The user is already authenticated. Use tools to "
                                  "look up orders; never guess."},
    {"role": "user", "content": "What is the status of my order #W4597054?"},
]
first = litellm.completion(messages=messages, tools=TOOLS, **KW)
msg = first.choices[0].message
print("turn 1 finish:", first.choices[0].finish_reason)
print("turn 1 tool_calls:", [(c.function.name, c.function.arguments) for c in msg.tool_calls or []])
assert msg.tool_calls, "expected a tool call"
messages.append(msg.model_dump(exclude_none=True))  # as agent frameworks append LiteLLM messages
messages.append({"role": "tool", "tool_call_id": msg.tool_calls[0].id,
                 "content": json.dumps({"order_id": "#W4597054", "status": "delivered",
                                        "items": [{"name": "Desk Lamp", "price": 39.99}]})})
second = litellm.completion(messages=messages, tools=TOOLS, **KW)
print("turn 2 finish:", second.choices[0].finish_reason)
print("turn 2 content:", second.choices[0].message.content)
print("usage:", second.usage.prompt_tokens, second.usage.completion_tokens)
