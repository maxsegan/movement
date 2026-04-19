#!/bin/bash
# Keeps vLLM server + scale_judge running for multi-day scale-out.
# vLLM dying restarts it; scale_judge dying restarts with --resume.
# Exits only on explicit kill (no clean exit case on finish — add one).

set -u
OUT_JSON=${OUT_JSON:-tests/scale_strat332k.json}
CLIP_LIST=${CLIP_LIST:-tests/full_325k.txt}
CONCURRENCY=${CONCURRENCY:-12}
CHUNK_SIZE=${CHUNK_SIZE:-48}
VLLM_URL=${VLLM_URL:-http://localhost:8000/v1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

vllm_alive() {
    curl -sf --max-time 5 "$VLLM_URL/models" > /dev/null 2>&1
}

wait_vllm_ready() {
    for _ in $(seq 1 180); do
        vllm_alive && return 0
        sleep 10
    done
    return 1
}

ensure_vllm() {
    if ! vllm_alive; then
        echo "[watchdog $(date -u +%H:%M:%S)] vLLM down, starting..."
        nohup bash "$REPO/scripts/serve_vlm.sh" > /tmp/vllm_watchdog.log 2>&1 &
        if ! wait_vllm_ready; then
            echo "[watchdog] vLLM failed to come up in 30 min; sleeping 5 min before retry"
            sleep 300
            return 1
        fi
        echo "[watchdog $(date -u +%H:%M:%S)] vLLM ready"
    fi
    return 0
}

while true; do
    if ! ensure_vllm; then
        continue
    fi
    # Check if done
    DONE=$(/usr/bin/python3.12 -c "
import json, sys
try:
    d = json.load(open('$OUT_JSON'))
    n = d['n']
except Exception:
    n = 0
total = sum(1 for _ in open('$CLIP_LIST'))
print(f'{n} {total}')
    " 2>/dev/null)
    set -- $DONE
    N=${1:-0}
    TOTAL=${2:-0}
    if [ "$N" -gt 0 ] && [ "$N" -ge "$TOTAL" ]; then
        echo "[watchdog $(date -u +%H:%M:%S)] All $N/$TOTAL clips judged. Exiting."
        exit 0
    fi
    echo "[watchdog $(date -u +%H:%M:%S)] launching scale_judge ($N/$TOTAL done)"
    VLM_SERVER_URL=$VLLM_URL /usr/bin/python3.12 "$REPO/scripts/scale_judge.py" \
        --clip-list "$CLIP_LIST" \
        --out "$OUT_JSON" \
        --concurrency "$CONCURRENCY" \
        --chunk-size "$CHUNK_SIZE" \
        --resume
    code=$?
    echo "[watchdog $(date -u +%H:%M:%S)] scale_judge exited code=$code; restarting in 60s"
    sleep 60
done
