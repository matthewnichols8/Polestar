"""Grow one conversation turn by turn (as an agent does, so prefix reuse applies) and record time and GPU memory per
turn, to find the largest context each server configuration can serve.  python context_probe.py --port 8011"""
import argparse, json, subprocess, time, urllib.request

p = argparse.ArgumentParser(); p.add_argument("--port", type=int, default=8011); p.add_argument("--max", type=int, default=32000)
p.add_argument("--chunk", type=int, default=4000); p.add_argument("--out"); a = p.parse_args()
src = open("/scratch/mi72/envs/swe/lib/python3.12/site-packages/swebench/harness/run_evaluation.py").read()
def gpu_mem():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [int(x) for x in o.split()]
msgs = [{"role": "system", "content": "You are a helpful assistant that can interact with a computer shell."}]
rows, pos, turn = [], 0, 0
while True:
    turn += 1
    piece = src[pos:pos + a.chunk * 3]; pos += a.chunk * 3
    if not piece: pos, piece = 0, src[:a.chunk * 3]
    msgs.append({"role": "user", "content": f"Here is part {turn} of a file:\n```\n{piece}\n```\nSummarize it in one sentence."})
    body = json.dumps({"model": "x", "messages": msgs, "max_tokens": 256}).encode()
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", body,
                                                                    {"content-type": "application/json"}), timeout=600))
    except Exception as e:  # noqa: BLE001
        rows.append({"turn": turn, "error": str(e)[:300], "gpu_mib": gpu_mem()}); print(rows[-1], flush=True); break
    g = r["dgemma"]; msgs.append({"role": "assistant", "content": r["choices"][0]["message"]["content"] or ""})
    rows.append({"turn": turn, "prompt_tokens": g["prompt_tokens"], "reused": g["reused_prompt_tokens"], "steps": g["steps"],
                 "seconds": round(time.time() - t, 2), "ms_per_step": round(1000 * g["seconds"] / max(g["steps"], 1)),
                 "gpu_mib": gpu_mem()})
    print(rows[-1], flush=True)
    if g["prompt_tokens"] >= a.max: break
if a.out: json.dump(rows, open(a.out, "w"), indent=1)
