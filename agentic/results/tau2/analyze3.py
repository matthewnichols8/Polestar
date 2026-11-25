"""Three-way comparison on retail tasks 0-19: dInfer baseline, Polestar CACHE, Polestar CACHE+COMMIT."""
import glob, json, statistics as st
SIM = "/scratch/mi72/tau2-bench/data/simulations"; R = "/scratch/mi72/agentic-dgemma/results/tau2"
TASKS = set(range(20))
RUNS = {
 "dgemma-dinfer-retail25": (["dgemma-dinfer-retail25"], ["dgemma-dinfer-retail25/server"]),
 "dgemma-polestar-cache-retail25": (["dgemma-polestar-cache-retail25"], ["dgemma-polestar-cache-retail25/server-part1"]),
 "dgemma-polestar-cache-commit-retail20": (["dgemma-polestar-cache-commit-retail20"], ["dgemma-polestar-cache-commit-retail20/server"]),
}
out = {}
for name, (sims_dirs, logs) in RUNS.items():
    sims = {}
    for s in sims_dirs:
        for x in json.load(open(f"{SIM}/{s}/results.json"))["simulations"]:
            if int(x["task_id"]) in TASKS: sims[int(x["task_id"])] = x
    reqs = [json.loads(l) for d in logs for f in glob.glob(f"{R}/{d}/requests.jsonl") for l in open(f)]
    reqs = [q for q in reqs if q["prompt_tokens"] > 1000]
    # part1 of the CACHE run covers tasks 0-19 exactly (it hung on task 20 before any reply was logged? no: keep all part1)
    steps = sum(q["steps"] for q in reqs); secs = sum(q["seconds"] for q in reqs)
    o = {"tasks": len(sims), "solved": int(sum(x["reward_info"]["reward"] for x in sims.values())),
         "rewards": "".join(str(int(sims[k]["reward_info"]["reward"])) for k in sorted(sims)),
         "user_cost_usd": round(sum(x.get("user_cost") or 0 for x in sims.values()), 3),
         "mean_task_seconds": round(st.mean(x["duration"] for x in sims.values()), 1),
         "requests": len(reqs), "malformed_calls": sum(q["malformed_calls"] for q in reqs),
         "mean_new_tokens": round(st.mean(q["new_tokens"] for q in reqs), 1),
         "steps": steps, "steps_per_request": round(steps / len(reqs), 2),
         "ms_per_step": round(1000 * secs / steps), "mean_request_seconds": round(secs / len(reqs), 2),
         "tokens_per_step": round(sum(q["new_tokens"] for q in reqs) / steps, 2)}
    cc = [c for q in reqs for c in (q.get("canvas_cache") or [])]
    if cc:
        o["cache_full_steps"] = sum(c["full_steps"] for c in cc); o["cache_partial_steps"] = sum(c["partial_steps"] for c in cc)
        o["mean_recompute_frac_partial"] = round(st.mean(f for c in cc for f in c["recompute_frac"]), 3)
        o["commit_refreshes"] = sum(c.get("commit_refreshes", 0) for c in cc)
    cm = [c for q in reqs for c in (q.get("commit") or [])]
    if cm:
        for k in ("base_accepted", "drift_commits", "extra_accepted", "frozen"): o["commit_" + k] = sum(c[k] for c in cm)
        o["requests_with_commit_stats"] = sum(1 for q in reqs if q.get("commit"))
    out[name] = o
json.dump(out, open(f"{R}/summary_tasks0-19.json", "w"), indent=2)
print(json.dumps(out, indent=1))
