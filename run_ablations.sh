#!/bin/bash

# This script demonstrates how to run ablation studies for the GraphHook.
# Each command runs a short, single-seed training run with a specific setting.

# --- Configuration ---
set -e

# Check for dataset path argument
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/your/dataset"
    exit 1
fi

DATA_PATH=$1
BASE_NAME="ablation"
SEED=42
OUTPUT_BASE_DIR="outputs_ablation"

# --- Shared training arguments for quick runs ---
TRAIN_ARGS=(
    --dataroot "$DATA_PATH"
    --seed "$SEED"
    --epochs 5 # Short run for demonstration
    --batch 8
    --eval-every-iters 1000
    --amp
)

echo "===================================================="
echo "--- RUNNING ABLATION STUDIES ---"
echo "--- Note: These are short demo runs. ---"
echo "===================================================="

# --- Ablation 1: Region Loss Only ---
echo "\n--- Ablation: Region Loss Only ---"
python train.py \
    "${TRAIN_ARGS[@]}" \
    --out "${OUTPUT_BASE_DIR}/${BASE_NAME}_region_only" \
    --graph on \
    --graph-edge-weight 0.0

# --- Ablation 2: Edge Loss Only ---
echo "\n--- Ablation: Edge Loss Only ---"
python train.py \
    "${TRAIN_ARGS[@]}" \
    --out "${OUTPUT_BASE_DIR}/${BASE_NAME}_edge_only" \
    --graph on \
    --graph-region-weight 0.0

# --- Ablation 3: Different number of segments ---
echo "\n--- Ablation: 96 Segments ---"
python train.py \
    "${TRAIN_ARGS[@]}" \
    --out "${OUTPUT_BASE_DIR}/${BASE_NAME}_segments96" \
    --graph on \
    --graph-segments 96

# --- Ablation 4: Different compactness ---
echo "\n--- Ablation: Compactness 16 ---"
python train.py \
    "${TRAIN_ARGS[@]}" \
    --out "${OUTPUT_BASE_DIR}/${BASE_NAME}_compactness16" \
    --graph on \
    --graph-compactness 16

# --- Ablation 5: Different graph weight ---
echo "\n--- Ablation: Graph Weight 0.2 ---"
python train.py \
    "${TRAIN_ARGS[@]}" \
    --out "${OUTPUT_BASE_DIR}/${BASE_NAME}_graphw0.2" \
    --graph on \
    --graph-w 0.2

echo "\n===================================================="
echo "--- ABLATION DEMO SCRIPT FINISHED ---"
echo "===================================================="
