#!/bin/bash
# Run DiffusionGemma on the 20 selected SWE-bench Verified instances in each server configuration, then score.
#   bash run_all.sh [label ...]     (default: all three labels, in order)
# Per label: start serve_openai.py with that configuration's flags, run mini-swe-agent, stop the server, score
# with eval_bwrap.py. If the server dies mid-run, the agent run is stopped, the server restarted and the agent run
# resumed (mini-swe-agent skips instances already in preds.json).
set -u
SWE=/scratch/mi72/agentic-dgemma/swe
SERVER=/scratch/mi72/agentic-dgemma/server
RESULTS=/scratch/mi72/agentic-dgemma/results/swe
export TMPDIR=/scratch/mi72/tmp HF_HOME=/scratch/mi72/hf
COMMON="--gpus 0,1 --port 8011 --dist-port 29711 --cache-slots 1 --default-max-tokens 1024 --max-prompt-tokens ${MAXP:-24000}"
declare -A FLAGS=(
  [dgemma-dinfer-swev20]=""
  [dgemma-polestar-cache-swev20]="--polestar --polestar-topk 2 --polestar-refresh-every 8"
  [dgemma-polestar-cache-commit-swev20]="--polestar --polestar-topk 2 --polestar-refresh-every 8 --polestar-commit"
)
LABELS=("$@"); [ ${#LABELS[@]} -eq 0 ] && LABELS=(dgemma-dinfer-swev20 dgemma-polestar-cache-swev20 dgemma-polestar-cache-commit-swev20)
FILTER=$(cat $SWE/filter.txt)
STALL=${STALL:-1200}   # seconds of server-log silence that count as a hang

server_pid() { pgrep -f "^/scratch/mi72/envs/polestar/bin/python serve_openai" | head -1; }
stop_server() {
  local s; s=$(server_pid); [ -z "$s" ] && return
  kill $s $(pgrep -P $s) 2>/dev/null; sleep 5; kill -9 $s $(pgrep -P $s) 2>/dev/null
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>500' | wc -l)" = 0 ]; do sleep 2; done
}
start_server() {  # $1 = label, $2 = attempt
  local out=$RESULTS/$1
  (cd $SERVER && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH= nohup /scratch/mi72/envs/polestar/bin/python \
     serve_openai.py $COMMON ${FLAGS[$1]} --log-dir $out/server > $out/server-$2.log 2>&1 &)
  until curl -sf -m 3 http://127.0.0.1:8011/health | grep -q ok; do
    sleep 5; [ -z "$(server_pid)" ] && { echo "server failed to start for $1"; return 1; }
  done
  return 0   # (without this the function returns the status of the last `[ -z ... ]` test, i.e. failure)
}

for label in "${LABELS[@]}"; do
  out=$RESULTS/$label; mkdir -p $out
  PC=$(git -C /scratch/mi72/Polestar rev-parse --short HEAD)
  cat > $out/config.json <<EOF
{"label": "$label", "model": "google/diffusiongemma-26B-A4B-it", "expert_dtype": "int8", "gpus": "2x TITAN RTX",
 "server_flags": "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $COMMON ${FLAGS[$label]}",
 "polestar_repo_commit": "$PC", "polestar_repo_branch": "$(git -C /scratch/mi72/Polestar branch --show-current)",
 "agent": "mini-swe-agent $(/scratch/mi72/envs/swe/bin/python -c 'import minisweagent;print(minisweagent.__version__)' 2>/dev/null | tail -1)",
 "agent_config": "swebench.yaml + $SWE/dgemma_swebench.yaml (step_limit 75, max_tokens 1024)",
 "environment": "fakesingularity (bwrap over unpacked swebench/sweb.eval images)",
 "benchmark": "SWE-bench Verified, 20 instances (seed 42, $SWE/instances.txt)", "scoring": "eval_bwrap.py (local)"}
EOF
  echo "=== $label start $(date -u +%FT%TZ)"
  stop_server
  attempt=1; while [ -e $out/server-$attempt.log ]; do attempt=$((attempt + 1)); done   # keep earlier logs
  while :; do
    start_server $label $attempt || break
    (cd $SWE && /scratch/mi72/envs/swe/bin/mini-extra swebench -c swebench.yaml -c $SWE/dgemma_swebench.yaml \
        --subset verified --split test --filter "$FILTER" -w 1 -o $out >> $out/agent.log 2>&1) &
    agent=$!
    crashed=0
    while kill -0 $agent 2>/dev/null; do
      if [ -z "$(server_pid)" ] || grep -q -E "OutOfMemoryError|Traceback" $out/server-$attempt.log; then
        crashed=1; echo "server crashed (attempt $attempt) $(date -u +%T); stopping agent"; pkill -P $agent; kill $agent; sleep 5; break
      fi
      # A hung server (ranks out of step in a collective) stays alive, so also restart when the server log has
      # been silent for $STALL s: every request logs a line, and no request or agent step takes anywhere near that.
      if [ $(( $(date +%s) - $(stat -c %Y $out/server-$attempt.log) )) -gt $STALL ]; then
        crashed=1; echo "server stalled (attempt $attempt) $(date -u +%T): no log output for ${STALL}s; dumping stacks, stopping agent"
        for p in $(pgrep -P $(server_pid)); do kill -USR1 $p 2>/dev/null; done; sleep 10
        cp $out/server-$attempt.log $out/server-$attempt.stall-stacks.log
        pkill -P $agent; kill $agent; sleep 5; break
      fi
      sleep 15
    done
    wait $agent 2>/dev/null
    stop_server
    [ $crashed = 0 ] && break
    attempt=$((attempt + 1)); [ $attempt -gt 6 ] && { echo "giving up on $label after repeated crashes"; break; }
  done
  echo "=== $label agent done $(date -u +%FT%TZ)"
  (cd $SWE && /scratch/mi72/envs/swe/bin/python eval_bwrap.py --preds $out/preds.json --instances $SWE/instances.txt \
       --workers 4 --out $out/eval 2>&1 | grep -E "RESOLVED|ERROR")
done
echo "=== all done $(date -u +%FT%TZ)"
