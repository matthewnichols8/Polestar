"""Analyze Polestar-Commit traces of DiffusionGemma (written by `evaluate_gsm8k_diffusion_gemma.py --commit record
--commit-trace-dir DIR`) and decide whether drift is a useful commit signal on its uniform-noise canvas.

A token's prediction at step t is "final" if it equals the token the canvas finally committed. Commit's premise is
that, among moderately confident tokens, a drift jump marks tokens whose prediction is already final. This script
measures that, sweeps candidate settings for drift-gated acceptance, reports the precision a persistent "freeze"
of drift commits would have, and simulates the confidence-threshold freeze (`--commit-freeze-threshold`): how
often a token's argmax at the first step its raw confidence reaches the threshold is the final token.

The last step of each canvas is excluded (its argmax *is* the final canvas), as are positions after the first EOS
(they are discarded).

    python integrations/dinfer/benchmarks/analyze_dgemma_commit_trace.py --trace-dir DIR --output REPORT.md
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np

CONF_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001]
FREEZE_THRESHOLDS = (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def load_canvases(trace_dir):
    canvases = []
    for path in sorted(glob.glob(os.path.join(trace_dir, "*.npz"))):
        t = dict(np.load(path))
        T = t["cur_step"].shape[0]
        if T < 2:
            continue   # nothing but the final step
        valid = np.zeros(t["final"].shape[0], dtype=bool)
        valid[: int(t["valid_len"])] = True
        t["mask"] = np.zeros((T, valid.shape[0]), dtype=bool)
        t["mask"][:-1] = valid          # drop the last step and positions after EOS
        t["argmax_final"] = t["argmax"] == t["final"][None]
        t["sampled_final"] = t["sampled"] == t["final"][None]
        t["name"] = os.path.basename(path)
        canvases.append(t)
    return canvases


def drift_delta(drift, window):
    """delta[t] = drift[t] - mean(drift over the previous `window` steps with drift); drift[t] on the first one.
    NaN where there is no drift (the first step)."""
    delta = np.full_like(drift, np.nan)
    history = []
    for t in range(drift.shape[0]):
        if np.isnan(drift[t]).all():
            continue
        delta[t] = drift[t] - np.mean(history, axis=0) if history else drift[t]
        history.append(drift[t])
        history = history[-window:]
    return delta


def commits(conf, delta, threshold, lower, gate_scale, cap):
    gate = gate_scale * np.clip(threshold - conf, 0.0, None) ** 2
    with np.errstate(invalid="ignore"):
        return (conf > lower) & (conf < threshold) & (delta >= gate) & (delta < cap)


def rate(hits, mask):
    n = int(mask.sum())
    return (float(hits[mask].mean()) if n else float("nan")), n


def pct(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


def analyze(canvases, threshold=0.9, cap=2.0):
    cat = lambda key: np.concatenate([c[key][c["mask"]] for c in canvases])  # noqa: E731
    report = {"canvases": len(canvases),
              "steps": int(sum(c["cur_step"].shape[0] for c in canvases)),
              "token_steps_analyzed": int(sum(c["mask"].sum() for c in canvases))}
    report["mean_steps_per_canvas"] = report["steps"] / max(1, len(canvases))

    # 1. how often a step's prediction is already final, by confidence
    by_conf = []
    for kind in ("conf_raw", "conf_tempered"):
        conf, am, sm = cat(kind), cat("argmax_final"), cat("sampled_final")
        for lo, hi in zip(CONF_BINS[:-1], CONF_BINS[1:]):
            m = (conf >= lo) & (conf < hi)
            a, n = rate(am, m)
            s, _ = rate(sm, m)
            by_conf.append({"confidence": kind, "bin": f"[{lo:.2f}, {min(hi, 1.0):.2f})", "n": n,
                            "argmax_final": a, "sampled_final": s})
    report["by_confidence"] = by_conf
    base = cat("base_accepted")
    report["base_accepted_per_step"] = float(np.mean([c["base_accepted"][:-1].sum(1).mean() for c in canvases]))
    report["base_accepted_sampled_final"], report["base_accepted_n"] = rate(cat("sampled_final"), base)

    # 2. Commit's premise and 3. the settings sweep
    sweep = []
    for kind, lower, window, gate_scale in itertools.product(
            ("conf_raw", "conf_tempered"), (0.5, 0.6, 0.7), (2, 3, 5), (5.0, 10.0, 20.0)):
        per = []
        for c in canvases:
            delta = drift_delta(c["drift"], window)
            conf = c[kind]
            band = (conf > lower) & (conf < threshold) & ~np.isnan(delta)
            sel = commits(conf, delta, threshold, lower, gate_scale, cap)
            per.append((band & c["mask"], sel & c["mask"], c))
        band = np.concatenate([b[c["mask"]] for b, _, c in per])
        sel = np.concatenate([s[c["mask"]] for _, s, c in per])
        base_c = np.concatenate([c["base_accepted"][c["mask"]] for _, _, c in per])
        am = np.concatenate([c["argmax_final"][c["mask"]] for _, _, c in per])
        sm = np.concatenate([c["sampled_final"][c["mask"]] for _, _, c in per])
        extra = sel & ~base_c
        no_event = band & ~sel
        steps = sum(int(c["mask"].any(axis=1).sum()) for _, _, c in per)
        row = {"confidence": kind, "lower_bound": lower, "window": window, "gate_scale": gate_scale,
               "commits": int(sel.sum()), "extra": int(extra.sum()),
               "extra_per_step": int(extra.sum()) / max(1, steps),
               "redundant_with_entropy_rule": float((sel & base_c).sum() / sel.sum()) if sel.any() else float("nan")}
        row["event_sampled_final"], _ = rate(sm, sel)
        row["event_argmax_final"], _ = rate(am, sel)
        row["no_event_sampled_final"], row["no_event_n"] = rate(sm, no_event)
        row["extra_sampled_final"], _ = rate(sm, extra)
        row["extra_argmax_final"], _ = rate(am, extra)
        sweep.append(row)
    report["sweep"] = sweep
    report["freeze"] = [simulate_freeze(canvases, tau) for tau in FREEZE_THRESHOLDS]
    return report


def simulate_freeze(canvases, threshold):
    """Freeze each valid position at the first step (before the last) its raw confidence reaches `threshold`, to that
    step's argmax. Precision = frozen token equals the final token. The traces come from unfrozen runs, so this is
    the precision of a freeze's first decision, ignoring how freezing would change later steps."""
    frozen = correct = 0
    for c in canvases:
        hit = (c["conf_raw"] >= threshold) & c["mask"]       # the mask drops the last step and post-EOS positions
        cols = np.nonzero(hit.any(axis=0))[0]
        first = hit[:, cols].argmax(axis=0)
        frozen += cols.size
        correct += int((c["argmax"][first, cols] == c["final"][cols]).sum())
    return {"threshold": threshold, "frozen": frozen, "frozen_per_canvas": frozen / max(1, len(canvases)),
            "precision": correct / frozen if frozen else float("nan"), "wrong": frozen - correct}


def to_markdown(r):
    out = ["# DiffusionGemma Polestar-Commit trace analysis", "",
           f"{r['canvases']} canvases, {r['steps']} denoising steps ({r['mean_steps_per_canvas']:.1f} per canvas), "
           f"{r['token_steps_analyzed']} token-steps analyzed (last step of each canvas and post-EOS positions "
           "excluded).", "",
           f"Entropy-bound sampler: {r['base_accepted_per_step']:.1f} positions accepted per step; their sampled "
           f"token is final {pct(r['base_accepted_sampled_final'])} of the time (n={r['base_accepted_n']}). "
           "This is the precision a drift commit should match.", "",
           "## 1. Is a step's prediction already final? (by confidence)", "",
           "| confidence | bin | n | argmax final | sampled final |", "|---|---|---|---|---|"]
    for b in r["by_confidence"]:
        out.append(f"| {b['confidence']} | {b['bin']} | {b['n']} | {pct(b['argmax_final'])} | "
                   f"{pct(b['sampled_final'])} |")
    out += ["", "## 2. Commit's premise: in the confidence band, does a drift event mark final tokens?", "",
            "Band = lower_bound < conf < 0.9 with drift available. *event* = the drift gate fires.", "",
            "| confidence | lower | window | gate | events | event: sampled final | no event: sampled final | "
            "extra (not entropy-accepted) | extra/step | extra: sampled final | extra: argmax final | "
            "redundant with entropy rule |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(r["sweep"], key=lambda s: (s["confidence"], s["lower_bound"], s["window"], s["gate_scale"])):
        out.append(f"| {s['confidence']} | {s['lower_bound']} | {s['window']} | {s['gate_scale']:g} | {s['commits']} | "
                   f"{pct(s['event_sampled_final'])} | {pct(s['no_event_sampled_final'])} | {s['extra']} | "
                   f"{s['extra_per_step']:.2f} | {pct(s['extra_sampled_final'])} | {pct(s['extra_argmax_final'])} | "
                   f"{pct(s['redundant_with_entropy_rule'])} |")
    out += ["", "## 3. Reading the tables", "",
            "* Premise holds if *event: sampled final* is clearly above *no event: sampled final* (same band).",
            "* Mode A (per-step acceptance) is worth trying for settings with a useful *extra/step* whose "
            "*extra: sampled final* is close to the entropy-bound acceptances' precision above.",
            "* A persistent freeze (mode B) needs *extra: argmax/sampled final* >= 98% and at least the entropy-bound "
            "precision; otherwise frozen errors cannot be corrected later in the canvas.", "",
            "## 4. Confidence-threshold freeze (simulated)", "",
            "Each position is frozen to its argmax at the first step its raw confidence reaches the threshold.", "",
            "| threshold | frozen | per canvas | precision | wrong |", "|---|---|---|---|---|"]
    for f in r["freeze"]:
        out.append(f"| {f['threshold']} | {f['frozen']} | {f['frozen_per_canvas']:.0f} | {pct(f['precision'])} | "
                   f"{f['wrong']} |")
    out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output", required=True,
                        help="markdown report; a .json with the numbers is written next to it")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--drift-cap", type=float, default=2.0)
    args = parser.parse_args(argv)
    canvases = load_canvases(args.trace_dir)
    if not canvases:
        raise SystemExit(f"no traces with at least two steps in {args.trace_dir}")
    report = analyze(canvases, threshold=args.threshold, cap=args.drift_cap)
    with open(args.output, "w") as f:
        f.write(to_markdown(report))
    with open(os.path.splitext(args.output)[0] + ".json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
