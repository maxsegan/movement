#!/bin/bash
# Auto-runs after Phase 2 retrack orchestrator exits:
#   1. Merge shard JSONs → phase2_results_all.json + phase2_retracked_ok.txt
#   2. Relaunch vLLM (TP=4)
#   3. Wait for vLLM ready
#   4. Run scale_judge_npz under watchdog (auto-resume on crash)
#   5. Run final prog post-pass on retracked NPZs joined with re-judge VLM
#
# Designed to be invoked by the auto-trigger that watches the retrack
# orchestrator PID. Logs everything to /tmp/phase2_followup.log.
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

LOG=/tmp/phase2_followup.log
echo "[$(date -u +%H:%M:%S)] === Phase 2 follow-up starting ===" | tee -a $LOG

# Step 1: merge
echo "[$(date -u +%H:%M:%S)] step 1: merging shard JSONs" | tee -a $LOG
/usr/bin/python3.12 scripts/merge_phase2_shards.py 2>&1 | tee -a $LOG

# Step 2: launch vLLM
echo "[$(date -u +%H:%M:%S)] step 2: launching vLLM" | tee -a $LOG
nohup bash scripts/serve_vlm.sh > /tmp/vllm_phase2_rejudge.log 2>&1 &
VLLM_PID=$!
echo "  vLLM PID=$VLLM_PID" | tee -a $LOG

# Step 3: wait for ready (max 30 min)
echo "[$(date -u +%H:%M:%S)] step 3: waiting for vLLM ready" | tee -a $LOG
for _ in $(seq 1 180); do
    if curl -sf --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[$(date -u +%H:%M:%S)] vLLM ready" | tee -a $LOG
        break
    fi
    sleep 10
done
if ! curl -sf --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "[$(date -u +%H:%M:%S)] vLLM failed to come up — aborting" | tee -a $LOG
    exit 1
fi

# Step 4: run scale_judge_npz under simple watchdog loop
echo "[$(date -u +%H:%M:%S)] step 4: running scale_judge_npz (watchdog loop)" | tee -a $LOG
TOTAL=$(wc -l < tests/phase2_retracked_ok.txt)
while true; do
    DONE=0
    if [ -f tests/phase2_rejudge.json ]; then
        DONE=$(/usr/bin/python3.12 -c "
import json
try: print(json.load(open('tests/phase2_rejudge.json'))['n'])
except: print(0)
" 2>/dev/null || echo 0)
    fi
    if [ "$DONE" -ge "$TOTAL" ]; then
        echo "[$(date -u +%H:%M:%S)] all $DONE/$TOTAL re-judged" | tee -a $LOG
        break
    fi
    echo "[$(date -u +%H:%M:%S)] launching scale_judge_npz ($DONE/$TOTAL)" | tee -a $LOG
    if ! curl -sf --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "[$(date -u +%H:%M:%S)] vLLM died, restarting..." | tee -a $LOG
        nohup bash scripts/serve_vlm.sh > /tmp/vllm_phase2_rejudge.log 2>&1 &
        for _ in $(seq 1 180); do
            if curl -sf --max-time 3 http://localhost:8000/v1/models > /dev/null 2>&1; then break; fi
            sleep 10
        done
    fi
    VLM_SERVER_URL=http://localhost:8000/v1 /usr/bin/python3.12 scripts/scale_judge_npz.py \
        --clip-list tests/phase2_retracked_ok.txt \
        --npz-dirs tests/phase2_npz_s1 tests/phase2_npz_s2 tests/phase2_npz_s3 tests/phase2_npz_s4 \
        --out tests/phase2_rejudge.json \
        --concurrency 12 --chunk-size 48 --resume 2>&1 | tee -a $LOG
    echo "[$(date -u +%H:%M:%S)] scale_judge_npz exited; sleeping 60s before retry" | tee -a $LOG
    sleep 60
done

# Step 5: composite final dataset verdict
echo "[$(date -u +%H:%M:%S)] step 5: building final composite" | tee -a $LOG
/usr/bin/python3.12 scripts/compose_final_dataset.py 2>&1 | tee -a $LOG

echo "[$(date -u +%H:%M:%S)] === Phase 2 follow-up complete ===" | tee -a $LOG
