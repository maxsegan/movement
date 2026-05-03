#!/bin/bash
# Launch VLA training on movement-strict-v2 across 4 GPUs (DDP via torchrun).
# Single-node, no inter-node coordination.
#
# Usage:
#   bash scripts/launch_train_v2.sh           # fresh run
#   CHECKPOINT=path/to/ckpt.pt bash scripts/launch_train_v2.sh   # resume
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

CONFIG="${CONFIG:-training/config_kinetics_v2.yaml}"
NPROC="${NPROC:-4}"
LOG="${LOG:-/tmp/train_v2.log}"

# Qwen 4B was downloaded to a local cache (HF_HOME below points there)
export HF_HOME="${HF_HOME:-/home/max/movement/models/qwen3-vl-4b-cache}"
# Tokenizers warning suppression
export TOKENIZERS_PARALLELISM=false
# Faster download fallback if needed
export HF_HUB_ENABLE_HF_TRANSFER=1

CKPT_ARG=""
if [ -n "${CHECKPOINT:-}" ]; then
    CKPT_ARG="--checkpoint $CHECKPOINT"
fi

echo "[$(date -u +%H:%M:%S)] launching DDP training: nproc=$NPROC config=$CONFIG"
echo "[$(date -u +%H:%M:%S)] log: $LOG"

cd training
nohup /usr/bin/python3.12 -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    train_vla.py --config "../$CONFIG" $CKPT_ARG > "$LOG" 2>&1 &
TRAIN_PID=$!
echo "TRAIN_PID=$TRAIN_PID"
echo "follow: tail -f $LOG"
