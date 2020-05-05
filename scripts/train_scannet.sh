#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

export BATCH_SIZE=${BATCH_SIZE:-9}
export MAX_ITER=${MAX_ITER:-120000}
export MODEL=${MODEL:-Res16UNet34C}
export DATASET=${DATASET:-ScannetVoxelization2cmDataset}

export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./outputs/$DATASET/$MODEL-b$BATCH_SIZE-$MAX_ITER-$2/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

python -m main \
    --log_dir $LOG_DIR \
    --dataset $DATASET \
    --model $MODEL \
    --lr 1e-1 \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --max_iter $MAX_ITER \
    --train_limit_numpoints 1200000 \
    --train_phase train \
    $3 2>&1 | tee -a "$LOG"
