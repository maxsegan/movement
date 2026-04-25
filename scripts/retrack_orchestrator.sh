#!/bin/bash
# Launch N retrack workers, one per GPU, against a candidate clip list.
# Each worker takes shard K/N and writes to its own out-dir + out-json.
# Resumable; safe to re-run.
#
# Usage:
#   bash scripts/retrack_orchestrator.sh tests/retrack_cands_all.txt phase2
#
# Args:
#   $1 = clip-list file (default: tests/retrack_cands_all.txt)
#   $2 = output prefix (default: phase2)

set -u
CLIP_LIST="${1:-tests/retrack_cands_all.txt}"
OUT_PREFIX="${2:-phase2}"
N_GPUS="${N_GPUS:-4}"
CHECKPOINT="${CHECKPOINT:-25}"

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

LOG_DIR=/tmp
mkdir -p "tests/${OUT_PREFIX}_npz_s1" "tests/${OUT_PREFIX}_npz_s2" \
         "tests/${OUT_PREFIX}_npz_s3" "tests/${OUT_PREFIX}_npz_s4"

PIDS=()
for K in $(seq 1 "$N_GPUS"); do
    GPU=$((K - 1))
    OUT_DIR="tests/${OUT_PREFIX}_npz_s${K}"
    OUT_JSON="tests/${OUT_PREFIX}_results_s${K}.json"
    LOG="${LOG_DIR}/${OUT_PREFIX}_s${K}.log"
    echo "[$(date -u +%H:%M:%S)] launching shard $K/$N_GPUS on GPU $GPU → $OUT_JSON"
    CUDA_VISIBLE_DEVICES=$GPU /usr/bin/python3.12 "$REPO/scripts/retrack_clips.py" \
        --clip-list "$CLIP_LIST" \
        --shard "$K/$N_GPUS" \
        --resume \
        --skip-vlm-judge \
        --checkpoint-every "$CHECKPOINT" \
        --out-dir "$OUT_DIR" \
        --out-json "$OUT_JSON" \
        --device "cuda:0" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    sleep 2
done

echo "Workers launched: ${PIDS[@]}"
echo "Logs in $LOG_DIR/${OUT_PREFIX}_s{1,2,3,4}.log"
echo "Tail with: tail -f /tmp/${OUT_PREFIX}_s*.log"
wait "${PIDS[@]}"
echo "[$(date -u +%H:%M:%S)] all retrack shards done"
