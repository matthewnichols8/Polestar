"""Summarize the tau2 retail runs (baseline vs Polestar CACHE) from tau2 results + server request logs."""
import glob, json, os, statistics as st
SIM = "/scratch/mi72/tau2-bench/data/simulations"
R = "/scratch/mi72/agentic-dgemma/results/tau2"
RUNS = {
    "dgemma-dinfer-retail25": {"sims": ["dgemma-dinfer-retail25"], "logs": ["dgemma-dinfer-retail25/server"]},
    "dgemma-polestar-cache-retail25": {
        "sims": ["dgemma-polestar-cache-retail25", "dgemma-polestar-cache-retail25-part2", "dgemma-polestar-cache-retail25-part3"],
        "logs": ["dgemma-polestar-cache-retail25/server-part1", "dgemma-polestar-cache-retail25/server-part2-hung",
                 "dgemma-polestar-cache-retail25/server-part3"]},
}
out = {}
for name, cfg in RUNS.items():
    sims = {}
    for s in cfg["sims"]:
        p = f"{SIM}/{s}/results.json"
        if os.path.exists(p):
            for x in json.load(open(p))["simulations"]:
                sims[int(x["task_id"])] = x          # later parts only add tasks not finished earlier
    reqs = []
    for l in cfg["logs"]:
        for f in glob.glob(f"{R}/{l}/requests.jsonl"):
            reqs += [json.loads(line) for line in open(f)]
    reqs = [q for q in reqs if q["prompt_tokens"] > 1000]  # drop my manual check requests
    agent_turns = [m for x in sims.values() for m in x["messages"] if m["role"] == "assistant"]
    steps = sum(q["steps"] for q in reqs); secs = sum(q["seconds"] for q in reqs)
    cc = [c for q in reqs for c in (q.get("canvas_cache") or [])]
    o = {
        "tasks": len(sims), "solved": sum(x["reward_info"]["reward"] for x in sims.values()),
        "rewards": {k: sims[k]["reward_info"]["reward"] for k in sorted(sims)},
        "user_cost_usd": round(sum(x.get("user_cost") or 0 for x in sims.values()), 3),
        "mean_task_seconds": round(st.mean(x["duration"] for x in sims.values()), 1),
        "mean_messages": round(st.mean(len(x["messages"]) for x in sims.values()), 1),
        "agent_turns": len(agent_turns),
        "agent_tool_call_turns": sum(1 for m in agent_turns if m.get("tool_calls")),
        "server_requests": len(reqs),
        "malformed_calls": sum(q["malformed_calls"] for q in reqs),
        "self_tool_response": sum(q["self_tool_response"] for q in reqs),
        "empty_replies": sum(1 for q in reqs if q["new_tokens"] == 0 or (q["finish_reason"] == "stop" and not q["raw"].replace("<|channel>thought\n<channel|>", "").strip())),
        "mean_prompt_tokens": round(st.mean(q["prompt_tokens"] for q in reqs)),
        "max_prompt_tokens": max(q["prompt_tokens"] for q in reqs),
        "prefix_reused_frac": round(sum(q["reused_prompt_tokens"] for q in reqs) / sum(q["prompt_tokens"] for q in reqs), 3),
        "mean_request_seconds": round(secs / len(reqs), 2),
        "steps": steps, "ms_per_step_incl_prefill": round(1000 * secs / steps),
        "new_tokens": sum(q["new_tokens"] for q in reqs),
    }
    if cc:
        o["cache_full_steps"] = sum(c["full_steps"] for c in cc)
        o["cache_partial_steps"] = sum(c["partial_steps"] for c in cc)
        o["mean_recompute_frac_partial"] = round(st.mean(fr_all), 3) if (fr_all := [f for c in cc for f in c["recompute_frac"]]) else None
        o["requests_with_cache_stats"] = sum(1 for q in reqs if q.get("canvas_cache"))
    out[name] = o
a, b = out["dgemma-dinfer-retail25"]["rewards"], out["dgemma-polestar-cache-retail25"]["rewards"]
out["disagreements"] = {k: [a.get(k), b.get(k)] for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
json.dump(out, open(f"{R}/summary.json", "w"), indent=2)
print(json.dumps(out, indent=2))
