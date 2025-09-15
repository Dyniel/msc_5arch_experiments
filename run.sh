#!/bin/bash

# This script runs a multi-seed, paired ablation study for the DCGAN.
# For each seed, it:
# 1. Trains a model without the graph regularizer (OFF).
# 2. Trains a model with the graph regularizer (ON).
# 3. Evaluates both models and saves the metrics.

# --- Configuration ---
# Stop on error
set -e

# Check for dataset path argument
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/train_dataset [/path/to/val_dataset]"
    exit 1
fi

# --- Main Settings ---
DATA_PATH=$1
VAL_PATH=${2:-$1} # Use train path if val path is not provided
BASE_NAME="histopathology_gan"
SEEDS="42 101 202 303 404" # As per requirement for >= 5 seeds
OUTPUT_BASE_DIR="outputs"
REPORTS_BASE_DIR="reports"

# --- Training Hyperparameters ---
EPOCHS=25
BATCH_SIZE=16
Z_DIM=128
LR_G=1e-4
LR_D=2e-4
N_CRITIC=1

# --- Evaluation Settings ---
N_GEN=1024 # Increased for more robust metrics

# --- GraphHook Settings ---
GRAPH_W=0.05
GRAPH_W_WARMUP=5000 # Warm up over 5k steps
GRAPH_CACHE=true # Enable caching by default for speed

# --- Loop over all seeds ---
for seed in $SEEDS; do
    echo "===================================================="
    echo "--- STARTING RUN FOR SEED: $seed ---"
    echo "===================================================="

    # --- Define output paths for this seed ---
    OUT_DIR_OFF="${OUTPUT_BASE_DIR}/${BASE_NAME}_s${seed}_off"
    OUT_DIR_ON="${OUTPUT_BASE_DIR}/${BASE_NAME}_s${seed}_on"
    REPORTS_DIR_SEED="${REPORTS_BASE_DIR}/s${seed}"
    mkdir -p "$REPORTS_DIR_SEED"

    # --- Shared training arguments ---
    TRAIN_ARGS=(
        --dataroot "$DATA_PATH"
        --dataroot-val "$VAL_PATH"
        --seed "$seed"
        --epochs "$EPOCHS"
        --batch "$BATCH_SIZE"
        --zdim "$Z_DIM"
        --n-gen "$N_GEN"
        --lr-g "$LR_G"
        --lr-d "$LR_D"
        --n-critic "$N_CRITIC"
        --amp
    )

    # --- Run 1: Graph OFF ---
    echo "--- Starting Training: Graph OFF (Seed: $seed) ---"
    python train.py \
        "${TRAIN_ARGS[@]}" \
        --out "$OUT_DIR_OFF" \
        --graph off
    echo "--- Training Finished: Graph OFF (Seed: $seed) ---"

    # --- Run 2: Graph ON ---
    echo "--- Starting Training: Graph ON (Seed: $seed) ---"
    GRAPH_ARGS=()
    if [ "$GRAPH_CACHE" = true ]; then
        GRAPH_ARGS+=(--graph-cache-real-stats)
    fi
    python train.py \
        "${TRAIN_ARGS[@]}" \
        --out "$OUT_DIR_ON" \
        --graph on \
        --graph-w "$GRAPH_W" \
        --graph-w-warmup-steps "$GRAPH_W_WARMUP" \
        "${GRAPH_ARGS[@]}"
    echo "--- Training Finished: Graph ON (Seed: $seed) ---"

    # --- Evaluation for this seed ---
    echo "--- Starting Evaluation (Seed: $seed) ---"

    # Evaluate Graph OFF run
    echo "Evaluating Graph OFF run..."
    python eval_metrics.py \
        --real "$VAL_PATH" \
        --fake "${OUT_DIR_OFF}/fakes_ema" \
        --out-json "${REPORTS_DIR_SEED}/${BASE_NAME}_off_metrics.json"

    # Evaluate Graph ON run
    echo "Evaluating Graph ON run..."
    python eval_metrics.py \
        --real "$VAL_PATH" \
        --fake "${OUT_DIR_ON}/fakes_ema" \
        --out-json "${REPORTS_DIR_SEED}/${BASE_NAME}_on_metrics.json"

    echo "--- Evaluation Finished (Seed: $seed) ---"
    echo "Metrics saved in the '${REPORTS_DIR_SEED}' directory."

done

echo "===================================================="
echo "--- ALL SEEDS PROCESSED SUCCESSFULLY ---"
echo "===================================================="
