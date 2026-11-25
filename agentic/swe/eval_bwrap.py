"""Score SWE-bench predictions locally, without Docker: the official images unpacked by `fakesingularity` + bwrap.

Mirrors swebench.harness.run_evaluation.run_instance (swebench 5.0.2): fresh copy of the instance image, apply the
patch with the harness's own GIT_APPLY_CMDS chain, run the harness's own eval script (test_spec.eval_script), then
grade with swebench.harness.grading.get_eval_report. Multimodal image assets are not supported (not needed for
SWE-bench Verified).

    python eval_bwrap.py --preds results/swe/<label>/preds.json --out results/swe/<label>/eval
    python eval_bwrap.py --gold  --instances instances.txt --out results/swe/check-gold     # must resolve all
    python eval_bwrap.py --empty --instances instances.txt --out results/swe/check-empty    # must resolve none
"""

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from datasets import load_dataset
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name
from swebench.harness.constants import APPLY_PATCH_FAIL, APPLY_PATCH_PASS, CONTAINER_PATCH_FILE
from swebench.harness.grading import get_eval_report
from swebench.harness.run_evaluation import GIT_APPLY_CMDS
from swebench.harness.utils import make_test_spec

SHIM = str(Path(__file__).resolve().parent / "fakesingularity")
TMP = Path(os.environ.get("TMPDIR", "/scratch/mi72/tmp"))


def sh(sandbox, command, timeout=None):
    """Run `command` in the sandbox as the image's root user in /testbed; returns (exit code, output, timed out)."""
    cmd = [SHIM, "exec", "--pwd", "/testbed", "--writable", str(sandbox), "/bin/bash", "-c", command]
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                           timeout=timeout)
        return r.returncode, r.stdout, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return -1, out, True


def evaluate(instance, patch, out_dir, timeout):
    iid = instance["instance_id"]
    d = out_dir / iid
    if (d / "report.json").is_file():
        return iid, json.loads((d / "report.json").read_text())
    d.mkdir(parents=True, exist_ok=True)
    spec = make_test_spec(instance)
    pred = {"instance_id": iid, "model_patch": patch, "model_name_or_path": "local"}
    sandbox = TMP / f"eval-{iid}-{uuid.uuid4().hex[:8]}"
    log = []
    try:
        subprocess.run([SHIM, "build", "--sandbox", str(sandbox),
                        "docker://" + get_swebench_docker_image_name(instance)], check=True)
        (d / "patch.diff").write_text(patch or "")
        shutil.copy(d / "patch.diff", str(sandbox) + CONTAINER_PATCH_FILE)
        applied = False
        for attempt, git_apply in enumerate(GIT_APPLY_CMDS):
            if attempt:
                sh(sandbox, "git checkout -- . ; git clean -fd")
            code, out, _ = sh(sandbox, f"{git_apply} {CONTAINER_PATCH_FILE}")
            if code == 0:
                log.append(f"{APPLY_PATCH_PASS}:\n{out}")
                applied = True
                break
            log.append(f"Failed to apply patch: {git_apply}")
        if not applied and sh(sandbox, f"git apply --check --reverse {CONTAINER_PATCH_FILE}")[0] == 0:
            applied = True
            log.append(f"{APPLY_PATCH_PASS}: verified already applied")
        if not applied:
            log.append(f"{APPLY_PATCH_FAIL}:\n{out}")
            report = {iid: {"patch_is_None": patch is None, "patch_exists": bool(patch), "patch_successfully_applied":
                            False, "resolved": False}}
        else:
            (Path(sandbox) / "eval.sh").write_text(spec.eval_script)
            (d / "eval.sh").write_text(spec.eval_script)
            start = time.time()
            code, out, timed_out = sh(sandbox, "/bin/bash /eval.sh", timeout=timeout)
            (d / "test_output.txt").write_text(out + (f"\n\nTimeout error: {timeout} seconds exceeded." if timed_out
                                                      else ""))
            log.append(f"Test runtime: {time.time() - start:.1f} s, exit {code}, timed out {timed_out}")
            report = get_eval_report(test_spec=spec, prediction=pred, test_log_path=str(d / "test_output.txt"),
                                     include_tests_status=True)
        (d / "report.json").write_text(json.dumps(report, indent=2))
        return iid, report
    finally:
        (d / "run.log").write_text("\n".join(log))
        subprocess.run(["chmod", "-R", "u+rwX", str(sandbox)], check=False, capture_output=True)
        shutil.rmtree(sandbox, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--preds", help="mini-swe-agent preds.json")
    g.add_argument("--gold", action="store_true", help="score the reference patches (sanity check)")
    g.add_argument("--empty", action="store_true", help="score empty patches (sanity check)")
    p.add_argument("--instances", help="instance ids, one per line (default: all in --preds)")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timeout", type=int, default=1800)
    args = p.parse_args()

    data = {r["instance_id"]: r for r in load_dataset("SWE-bench/SWE-bench_Verified", split="test")}
    if args.preds:
        preds = json.loads(Path(args.preds).read_text())
        patches = {iid: (v.get("model_patch") or "") for iid, v in preds.items()}
    if args.instances:
        ids = [l.strip() for l in open(args.instances) if l.strip()]
    else:
        ids = sorted(patches)
    if args.gold:
        patches = {iid: data[iid]["patch"] for iid in ids}
    elif args.empty:
        patches = {iid: "" for iid in ids}
    patches = {iid: patches.get(iid, "") for iid in ids}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    with cf.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(evaluate, data[iid], patches[iid], out, args.timeout): iid for iid in ids}
        for f in cf.as_completed(futures):
            iid = futures[f]
            try:
                _, report = f.result()
                results[iid] = bool(report[iid]["resolved"])
            except Exception as e:  # noqa: BLE001
                results[iid] = None
                print(f"{iid}: ERROR {e}", file=sys.stderr)
            print(f"{iid}: resolved={results[iid]}", flush=True)
    summary = {"resolved": sum(1 for v in results.values() if v), "total": len(ids),
               "errors": [k for k, v in results.items() if v is None],
               "resolved_ids": sorted(k for k, v in results.items() if v), "results": results}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"RESOLVED {summary['resolved']}/{summary['total']}  errors={len(summary['errors'])}")


if __name__ == "__main__":
    main()
