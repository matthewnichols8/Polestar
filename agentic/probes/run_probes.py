"""Phase 0: run the tool-call probes through dInfer DiffusionGemma and score them.

Every probe is rendered with the checkpoint's chat template (Gemma's native tool syntax), generated with
CanvasDiffusionLLM on the expert-parallel model, and scored against its `expect` block. Each config in CONFIGS
runs every probe for every seed.

    PYTHONPATH= HF_HOME=/scratch/mi72/hf TMPDIR=/scratch/mi72/tmp \
    /scratch/mi72/envs/polestar/bin/python run_probes.py --gpus 0,1 --expert-dtype int8 --out ../results/phase0
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
PROBES = HERE / "tool_call_probes.jsonl"
MODEL_ID = "google/diffusiongemma-26B-A4B-it"

CONFIGS = {
    "base":      dict(entropy_bound=0.1, max_denoising_steps=48, thinking=False, polestar=False, gen_length=512),
    "thinking":  dict(entropy_bound=0.1, max_denoising_steps=48, thinking=True, polestar=False, gen_length=1024),
    "eb0.05":    dict(entropy_bound=0.05, max_denoising_steps=48, thinking=False, polestar=False, gen_length=512),
    "steps96":   dict(entropy_bound=0.1, max_denoising_steps=96, thinking=False, polestar=False, gen_length=512),
    "polestar":  dict(entropy_bound=0.1, max_denoising_steps=48, thinking=False, polestar=True, gen_length=512),
}

FILLER = ("Policy detail {i}: when handling customer requests, verify the account, check the order status, explain "
          "any fees or refund timelines precisely, never invent order ids, payment methods or item ids, and keep a "
          "polite, concise tone. Refunds go to the original payment method within 5 to 7 business days. ")

TOOL_CALL_ID, TOOL_CALL_END_ID, TOOL_RESPONSE_ID, TURN_END_ID, EOS_ID = 48, 49, 50, 106, 1
STR = '<|"|>'


# ---------------------------------------------------------------------------------------------- parsing
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


def parse_output(text: str):
    """Split generated text into thinking, visible text and tool calls."""
    thinking = "".join(re.findall(r"<\|channel>(.*?)<channel\|>", text, flags=re.S))
    visible = re.sub(r"<\|channel>.*?(<channel\|>|$)", "", text, flags=re.S)
    calls, errors = [], []
    for match in re.finditer(r"<\|tool_call>(.*?)(<tool_call\|>|$)", visible, flags=re.S):
        raw, closed = match.group(1), bool(match.group(2))
        m = re.match(r"\s*call:([A-Za-z_][A-Za-z0-9_\-]*)\s*(\{.*\})\s*$", raw, flags=re.S)
        if not closed or not m:
            errors.append(raw)
            continue
        try:
            calls.append({"name": m.group(1), "arguments": gemma_args_to_python(m.group(2))})
        except (ValueError, json.JSONDecodeError):
            errors.append(raw)
    reply = re.sub(r"<\|tool_call>.*?(<tool_call\|>|$)", "", visible, flags=re.S)
    reply = re.sub(r"<[|]?[a-z_\"]+[|]?>", "", reply).strip()
    return {"thinking": thinking, "reply": reply, "calls": calls, "call_errors": errors,
            "attempted_call": "<|tool_call>" in visible}


# ---------------------------------------------------------------------------------------------- scoring
def _norm(v):
    return json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v).strip().lower()


def score(expect: dict, parsed: dict) -> tuple[bool, str]:
    calls, reply = parsed["calls"], parsed["reply"].lower()
    names = [c["name"] for c in calls]
    if any(n in expect.get("forbidden_calls", []) for n in names):
        return False, f"forbidden call {names}"

    def text_ok():
        if any(s.lower() not in reply for s in expect.get("text_contains", [])):
            return False
        any_of = expect.get("text_contains_any")
        return not any_of or any(s.lower() in reply for s in any_of)

    kind = expect["type"]
    if kind == "no_call":
        if parsed["attempted_call"]:
            return False, f"unexpected call {names or parsed['call_errors']}"
        return (True, "ok") if text_ok() else (False, "reply missing expected text")
    if kind == "no_call_or_call":
        if parsed["attempted_call"]:
            if not calls:
                return False, "malformed call"
            ok = calls[0]["name"] in expect.get("allowed_calls", [])
            return ok, "ok" if ok else f"call {names[0]} not allowed"
        return (True, "ok") if text_ok() else (False, "reply missing expected text")
    if kind != "call":
        raise ValueError(kind)
    if not calls:
        return False, "malformed call" if parsed["attempted_call"] else "no call"
    call = calls[0]
    if call["name"] != expect["name"]:
        return False, f"wrong tool {call['name']}"
    args = call["arguments"]
    for key, value in expect.get("arguments", {}).items():
        if key not in args or _norm(args[key]) != _norm(value):
            return False, f"argument {key}={args.get(key)!r}"
    for key, needles in expect.get("arguments_contains", {}).items():
        if key not in args or any(n.lower() not in str(args[key]).lower() for n in needles):
            return False, f"argument {key}={args.get(key)!r}"
    command_any = expect.get("command_any")
    if command_any and not any(c in str(args.get("command", "")) for c in command_any):
        return False, f"command {args.get('command')!r}"
    return True, "ok"


# ---------------------------------------------------------------------------------------------- generation
def render(tokenizer, probe, thinking):
    messages = [dict(m) for m in probe["messages"]]
    pad = probe.get("pad_system_tokens")
    if pad:
        filler, i = "", 0
        while len(tokenizer(filler)["input_ids"]) < pad:
            filler += FILLER.format(i=i)
            i += 1
        messages[0]["content"] = messages[0]["content"] + "\n\n" + filler
    continues_turn = messages[-1]["role"] == "tool"
    return tokenizer.apply_chat_template(messages, tools=probe["tools"], add_generation_prompt=not continues_turn,
                                         tokenize=True, enable_thinking=thinking)


def stop_reason(decoder, prompt_len, out_len, canvas_length, finished):
    if not finished:
        return "budget"
    last_block_start = prompt_len + ((out_len - prompt_len) // canvas_length) * canvas_length
    token = int(decoder.argmax[0, out_len - last_block_start])
    return {TOOL_RESPONSE_ID: "self_tool_response", TURN_END_ID: "turn_end", EOS_ID: "eos"}.get(token, f"token_{token}")


def worker(rank, gpus, args):
    device = torch.device("cuda", gpus[rank])
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{args.port}", rank=rank,
                            world_size=len(gpus), device_id=device)
    from huggingface_hub import snapshot_download
    from dinfer import BlockIteratorFactory, CanvasDiffusionLLM, EntropyBoundParallelDecoder, KVCacheFactory
    from dinfer.model import DiffusionGemmaModelLM, load_diffusion_gemma_tokenizer
    from polestar.integrations.dinfer_canvas_cache import CanvasPolestarCache

    path = snapshot_download(MODEL_ID)
    model = DiffusionGemmaModelLM.from_pretrained(path, device=device, expert_dtype=args.expert_dtype)
    tokenizer = load_diffusion_gemma_tokenizer(path)
    probes = [json.loads(line) for line in PROBES.read_text().splitlines() if line.strip()]
    if args.only:
        probes = [p for p in probes if any(p["id"].startswith(o) for o in args.only.split(","))]
    configs = {k: v for k, v in CONFIGS.items() if not args.configs or k in args.configs.split(",")}
    out = Path(args.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        raw_file = (out / "raw.jsonl").open("w")
        print(model.placement_summary(), flush=True)
    rows = []
    for config_name, cfg in configs.items():
        for seed in range(args.seeds):
            decoder = EntropyBoundParallelDecoder(model.config.vocab_size, entropy_bound=cfg["entropy_bound"],
                                                  max_denoising_steps=cfg["max_denoising_steps"], seed=seed,
                                                  device=device)
            cache = CanvasPolestarCache(len(model.layers), topk=2, refresh_every=8, polestar_seed=seed) \
                if cfg["polestar"] else None
            llm = CanvasDiffusionLLM(model, decoder, BlockIteratorFactory(), KVCacheFactory("prefix", per_layer=True),
                                     canvas_cache=cache)
            for probe in probes:
                ids = torch.tensor([render(tokenizer, probe, cfg["thinking"])], device=device)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                output = llm.generate(ids, gen_length=cfg["gen_length"])
                torch.cuda.synchronize(device)
                seconds = time.perf_counter() - start
                if rank != 0:
                    continue
                stats = llm.last_stats
                text = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=False)
                parsed = parse_output(text)
                ok, why = score(probe["expect"], parsed)
                row = {
                    "config": config_name, "seed": seed, "probe": probe["id"], "level": probe["level"],
                    "correct": ok, "reason": why,
                    "attempted_call": parsed["attempted_call"],
                    "format_ok": (not parsed["attempted_call"]) or (bool(parsed["calls"]) and not parsed["call_errors"]),
                    "stop": stop_reason(decoder, ids.shape[1], output.shape[1], model.config.canvas_length,
                                        stats["finished"]),
                    "calls": parsed["calls"], "reply": parsed["reply"][:400], "thinking_chars": len(parsed["thinking"]),
                    "prompt_tokens": ids.shape[1], "new_tokens": stats["new_tokens"],
                    "steps": stats["num_forwards"], "seconds": round(seconds, 2), "raw": text,
                }
                rows.append(row)
                raw_file.write(json.dumps(row) + "\n")
                raw_file.flush()
                print(f"[{config_name} s{seed}] {probe['id']:24s} {'PASS' if ok else 'FAIL'}  {why:32s} "
                      f"stop={row['stop']:18s} {row['steps']:3d} steps {seconds:5.1f}s", flush=True)
    if rank == 0:
        raw_file.close()
        summarize(rows, out)
    dist.destroy_process_group()


def summarize(rows, out):
    table = defaultdict(lambda: defaultdict(list))
    for r in rows:
        table[r["config"]]["all"].append(r)
        table[r["config"]][f"L{r['level']}"].append(r)
    levels = sorted({k for t in table.values() for k in t if k != "all"})
    summary = {}
    lines = ["| config | correct | " + " | ".join(levels) + " | format ok (calls) | self tool_response | s/probe |",
             "|---" * (len(levels) + 5) + "|"]
    for config, groups in table.items():
        allr = groups["all"]
        attempted = [r for r in allr if r["attempted_call"]]
        entry = {
            "correct": sum(r["correct"] for r in allr) / len(allr),
            "by_level": {lv: sum(r["correct"] for r in groups[lv]) / len(groups[lv]) for lv in levels if groups[lv]},
            "format_ok_given_call": (sum(r["format_ok"] for r in attempted) / len(attempted)) if attempted else None,
            "self_tool_response": sum(r["stop"] == "self_tool_response" for r in allr) / len(allr),
            "seconds_per_probe": sum(r["seconds"] for r in allr) / len(allr),
            "n": len(allr),
        }
        summary[config] = entry
        fmt = lambda v: "n/a" if v is None else f"{v:.0%}"
        lines.append(f"| {config} | {fmt(entry['correct'])} | "
                     + " | ".join(fmt(entry["by_level"].get(lv)) for lv in levels)
                     + f" | {fmt(entry['format_ok_given_call'])} | {fmt(entry['self_tool_response'])}"
                     f" | {entry['seconds_per_probe']:.1f} |")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--expert-dtype", choices=["bf16", "int8"], default="int8")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--configs", help="comma-separated subset of CONFIGS")
    parser.add_argument("--only", help="comma-separated probe id prefixes")
    parser.add_argument("--out", default=str(HERE.parent / "results" / "phase0"))
    parser.add_argument("--port", type=int, default=29641)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    torch.multiprocessing.spawn(worker, args=(gpus, args), nprocs=len(gpus))
