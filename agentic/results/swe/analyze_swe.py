"""Summarise the SWE-bench runs: outcomes, tool-call reliability, speed, CACHE/COMMIT stats -> swe_summary.json."""
import ast, glob, json, statistics as st, sys
from pathlib import Path
R = Path(__file__).resolve().parent
LABELS = ["dgemma-dinfer-swev20", "dgemma-polestar-cache-swev20", "dgemma-polestar-cache-commit-swev20"]
out = {}
for l in LABELS:
    d = R / l
    preds = json.loads((d / "preds.json").read_text())
    ev = json.loads((d / "eval/summary.json").read_text())
    exits, steps = {}, []
    for iid in preds:
        t = json.loads(next((d / iid).glob("*.traj.json")).read_text())
        exits[iid] = t["info"].get("exit_status")
        steps.append(sum(1 for m in t["messages"] if m.get("role") == "assistant"))
    reqs = [json.loads(x) for x in open(d / "server/requests.jsonl")]
    def speed(rs):
        S = sum(r["steps"] for r in rs); T = sum(r["seconds"] for r in rs)
        warm = [r for r in rs if r["reused_prompt_tokens"] > 0]
        return {"requests": len(rs), "ms_per_step": round(1000 * T / S, 1), "steps_per_request": round(S / len(rs), 2),
                "s_per_request": round(T / len(rs), 2),
                "reuse_share": round(sum(r["reused_prompt_tokens"] for r in rs) / sum(r["prompt_tokens"] for r in rs), 3),
                "warm_requests": len(warm)}
    full = part = refresh = 0; frac = []
    commit_extra = commit_frozen = 0
    for r in reqs:
        cc = r.get("canvas_cache")
        if cc:
            cc = ast.literal_eval(cc) if isinstance(cc, str) else cc
            for c in cc:
                full += c["full_steps"]; part += c["partial_steps"]; frac += c.get("recompute_frac", [])
                refresh += c.get("commit_refreshes", 0)
        cm = r.get("commit")
        if cm:
            for c in (cm if isinstance(cm, list) else [cm]):
                if isinstance(c, dict):
                    commit_extra += c.get("extra_accepted", 0) or 0; commit_frozen += c.get("frozen", 0) or 0
    out[l] = {
        "resolved": ev["resolved"], "resolved_ids": ev["resolved_ids"],
        "submitted_patches": sum(1 for v in preds.values() if (v.get("model_patch") or "").strip()),
        "exit_status": {k: list(exits.values()).count(k) for k in sorted(set(exits.values()))},
        "agent_steps_median": st.median(steps), "agent_steps_mean": round(st.mean(steps), 1),
        "malformed_call_rate": round(sum(r.get("malformed_calls", 0) > 0 for r in reqs) / len(reqs), 3),
        "max_prompt_tokens_seen": max(r["prompt_tokens"] for r in reqs),
        "speed_all": speed(reqs),
        "speed_prompt_le_18k": speed([r for r in reqs if r["prompt_tokens"] <= 18000]),
        "cache": {"full_steps": full, "partial_steps": part,
                  "mean_recompute_frac_partial": round(st.mean(frac), 3) if frac else None,
                  "commit_refreshes": refresh} if full else None,
        "commit": {"extra_accepted": commit_extra, "frozen": commit_frozen} if commit_extra or commit_frozen else None,
    }
(R / "swe_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
