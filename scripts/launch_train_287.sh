#!/bin/bash
# Scheduled-launch wrapper for movement-287 training.
#
# Steps:
#   1. Package vlm-good NPZ if not already done (~30 min)
#   2. Wait for any leftover GPU users to clear
#   3. Launch DDP training on 4 GPUs
#
# Designed to be invoked via cron at 9am ET. All output to /tmp/launch_287.log.

set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

LOG=/tmp/launch_287.log
{
echo "=== launch_train_287 starting $(date -u +'%Y-%m-%d %H:%M UTC') ==="

NPZ_DIR="$REPO/data/kinetics_v2_vlm_good"
DESC_DIR="$REPO/data/kinetics_v2_vlm_good_desc"

# Step 1: NPZ packaging if needed
N_EXPECTED=286890
if [ -d "$NPZ_DIR" ]; then
    N_HAVE=$(find "$NPZ_DIR" -name '*.npz' | wc -l)
else
    N_HAVE=0
fi
echo "NPZ count: have=$N_HAVE expected=$N_EXPECTED"
if [ "$N_HAVE" -lt "$N_EXPECTED" ]; then
    echo "[$(date -u +%H:%M:%S)] packaging vlm-good NPZ..."
    /usr/bin/python3.12 "$REPO/scripts/package_v2.py" \
        --format npz --filter vlm_good --workers 18 2>&1 | tail -25
    N_HAVE=$(find "$NPZ_DIR" -name '*.npz' | wc -l)
    echo "[$(date -u +%H:%M:%S)] packaged: $N_HAVE NPZs"
fi

# Step 2: ensure GPUs free
for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
    if [ "${used:-0}" -lt 10000 ]; then break; fi
    echo "[$(date -u +%H:%M:%S)] waiting for GPUs (used=${used} MiB)"
    sleep 30
done

# Step 3: launch training
echo "[$(date -u +%H:%M:%S)] launching DDP training..."
CONFIG=training/config_kinetics_v2_287.yaml \
LOG=/tmp/train_287.log \
    bash "$REPO/scripts/launch_train_v2.sh"

echo "=== launch_train_287 finished $(date -u +'%Y-%m-%d %H:%M UTC') ==="
} >> "$LOG" 2>&1
