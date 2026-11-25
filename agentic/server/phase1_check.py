"""Phase 1 checks against a running serve_openai.py, through the OpenAI Python client.

  A. probes:    the 15 Phase 0 tool-call probes, scored as in Phase 0 (should match Phase 0's base config).
  B. multiturn: a scripted SWE-style conversation growing by real source files each turn, timed with prefix
                reuse and with `cache_reuse: false` (same prompts both times; the model's replies are discarded).
  C. api:       streaming, tool_choice="none", bad requests.

    PYTHONPATH= /scratch/mi72/envs/polestar/bin/python phase1_check.py --out ../results/phase1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import openai

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "probes"))
from run_probes import FILLER, score  # noqa: E402

SOURCE_FILES = sorted(Path("/scratch/mi72/Polestar/src/polestar/models/llada").glob("*.py"))


def _parsed(message):
    calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)}
             for c in message.tool_calls or []]
    reply = message.content or ""
    return {"calls": calls, "reply": reply, "attempted_call": bool(calls) or "<|tool_call>" in reply,
            "call_errors": []}


def check_probes(client, out, seeds):
    probes = [json.loads(l) for l in (HERE.parent / "probes" / "tool_call_probes.jsonl").read_text().splitlines()
              if l.strip()]
    rows = []
    for seed in range(seeds):
        for probe in probes:
            messages = [dict(m) for m in probe["messages"]]
            if probe.get("pad_system_tokens"):
                # ~4 characters per token, as run_probes pads to a token count with the tokenizer.
                filler, i = "", 0
                while len(filler) < 4 * probe["pad_system_tokens"]:
                    filler += FILLER.format(i=i)
                    i += 1
                messages[0]["content"] += "\n\n" + filler
            start = time.perf_counter()
            r = client.chat.completions.create(model="diffusiongemma", messages=messages, tools=probe["tools"],
                                               max_tokens=512, seed=seed)
            seconds = time.perf_counter() - start
            parsed = _parsed(r.choices[0].message)
            ok, why = score(probe["expect"], parsed)
            diag = r.model_extra.get("dgemma", {})
            rows.append({"seed": seed, "probe": probe["id"], "correct": ok, "reason": why,
                         "finish_reason": r.choices[0].finish_reason, "calls": parsed["calls"],
                         "reply": parsed["reply"][:300], "seconds": round(seconds, 2), **diag})
            print(f"[probes s{seed}] {probe['id']:24s} {'PASS' if ok else 'FAIL'} {why:28s} "
                  f"finish={r.choices[0].finish_reason:10s} reused={diag.get('reused_prompt_tokens')} "
                  f"{seconds:5.1f}s", flush=True)
    (out / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    correct = sum(r["correct"] for r in rows)
    print(f"[probes] {correct}/{len(rows)} correct", flush=True)
    return {"correct": correct, "n": len(rows),
            "finish_reasons": {k: sum(r["finish_reason"] == k for r in rows) for k in {r["finish_reason"] for r in rows}},
            "failures": [(r["seed"], r["probe"], r["reason"]) for r in rows if not r["correct"]]}


def _swe_conversation(turns):
    """Turn k's prompt = system + task + k scripted (assistant command, file contents) exchanges."""
    messages = [
        {"role": "system", "content": "You are a software engineer working in a bash shell. Reply with a short "
                                      "THOUGHT and exactly one ```bash``` block."},
        {"role": "user", "content": "Find where the k-means clustering is seeded and explain how to make it "
                                    "deterministic. Start by reading the source files."},
    ]
    prompts = [list(messages)]
    for path in SOURCE_FILES[:turns - 1]:
        text = path.read_text()[:6000]
        messages = messages + [
            {"role": "assistant", "content": f"THOUGHT: read the next file.\n```bash\ncat {path.name}\n```"},
            {"role": "user", "content": f"<returncode>0</returncode>\n<output>\n{text}\n</output>"},
        ]
        prompts.append(list(messages))
    return prompts


def check_multiturn(client, out, turns):
    prompts = _swe_conversation(turns)
    results = {}
    for mode, reuse in [("reuse", True), ("no_reuse", False)]:
        rows = []
        for k, messages in enumerate(prompts):
            start = time.perf_counter()
            r = client.chat.completions.create(model="diffusiongemma", messages=messages, max_tokens=256, seed=0,
                                               extra_body={"cache_reuse": reuse})
            seconds = time.perf_counter() - start
            d = r.model_extra["dgemma"]
            rows.append({"turn": k, "seconds": round(seconds, 2), **{key: d[key] for key in (
                "prompt_tokens", "reused_prompt_tokens", "common_prefix_tokens", "new_tokens", "steps")},
                "reply": (r.choices[0].message.content or "")[:200]})
            print(f"[multiturn {mode:8s}] turn {k:2d} prompt={d['prompt_tokens']:6d} "
                  f"reused={d['reused_prompt_tokens']:6d} steps={d['steps']:3d} {seconds:6.2f}s", flush=True)
        results[mode] = rows
    (out / "multiturn.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def check_api(client, base_url):
    import urllib.error
    import urllib.request

    report = {}
    stream = client.chat.completions.create(model="diffusiongemma", stream=True, max_tokens=64,
                                            messages=[{"role": "user", "content": "Say hello in three words."}])
    chunks = list(stream)
    report["stream_text"] = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    report["stream_finish"] = [c.choices[0].finish_reason for c in chunks if c.choices][-1]
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city.",
                                               "parameters": {"type": "object",
                                                              "properties": {"city": {"type": "string"}}}}}]
    r = client.chat.completions.create(model="diffusiongemma", tools=tools, tool_choice="none", max_tokens=128,
                                       messages=[{"role": "user", "content": "What's the weather in Paris?"}])
    report["tool_choice_none_has_calls"] = bool(r.choices[0].message.tool_calls)
    for name, body in [("n=2", {"messages": [{"role": "user", "content": "hi"}], "n": 2}),
                       ("bad args", {"messages": [{"role": "assistant", "content": None, "tool_calls": [
                           {"id": "1", "type": "function", "function": {"name": "f", "arguments": "{nope"}}]}]})]:
        req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req)
            report[name] = "accepted (unexpected)"
        except urllib.error.HTTPError as exc:
            report[name] = exc.code
    print("[api]", json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--out", default=str(HERE.parent / "results" / "phase1"))
    p.add_argument("--checks", default="api,probes,multiturn")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--turns", type=int, default=9)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(base_url=args.base_url, api_key="x", timeout=1800)
    summary = {}
    checks = args.checks.split(",")
    if "api" in checks:
        summary["api"] = check_api(client, args.base_url)
    if "probes" in checks:
        summary["probes"] = check_probes(client, out, args.seeds)
    if "multiturn" in checks:
        mt = check_multiturn(client, out, args.turns)
        summary["multiturn"] = {mode: {"total_seconds": round(sum(r["seconds"] for r in rows), 1),
                                       "last_turn_seconds": rows[-1]["seconds"],
                                       "last_prompt_tokens": rows[-1]["prompt_tokens"]}
                                for mode, rows in mt.items()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
