#!/usr/bin/env bash
# EXPERIMENTS (≈12h on A5000): StyleGAN2-ADA & DCGAN — with/without Graph Regularizer
# Run:  bash run_experiments.sh
export PYTHONWARNINGS="ignore::UserWarning"
set -euo pipefail

# -----------------------------
# CONFIG — EDIT THESE 3 LINES
# -----------------------------
GPU=0
SG2_DATA="vendors/style2_datasets/lung_scc256.zip"             # StyleGAN2 dataset (zip or dir)
DC_DATA="/home/student2/histo/data/lung_colon_image_set/lung_image_sets/lung_scc"  # DCGAN dataset (dir)
OUT="runs_experiments_12h"                                     # Root output folder

# Seeds and training sizes (12h preset)
SEEDS=(42)               # 1 seed dla porównywalności w ~12h
KIMG=1000                 # StyleGAN2 training length (thousands of images)
EPOCHS=100                # DCGAN training length (epochs)
BATCH_DC=64              # DCGAN batch size

# Perf knobs (safe defaults)
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

mkdir -p "${OUT}"

run_style2_baseline () {
  local seed="$1"
  local od="${OUT}/style2_baseline_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style2/train.py \
    --outdir "${od}" \
    --data "${SG2_DATA}" \
    --gpus 1 --cfg paper256 --mirror 1 \
    --aug ada --target 0.6 \
    --gamma 10 --seed "${seed}" \
    --kimg "${KIMG}" \
    --snap 5 \
    --metrics fid50k \
    | tee "${od}/console.log"
}

run_style2_graph () {
  local seed="$1"
  local od="${OUT}/style2_graph_diff_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style2/train.py \
    --outdir "${od}" \
    --data "${SG2_DATA}" \
    --gpus 1 --cfg paper256 --mirror 1 \
    --aug ada --target 0.6 \
    --gamma 10 --seed "${seed}" \
    --kimg "${KIMG}" \
    --snap 5 \
    --metrics fid50k \
    --graph \
    --graph-backend diff \
    --graph-w 1e-3 \
    --graph-subb 0 \
    --graph-bins 16 --graph-cells 8 --graph-down 4 \
    | tee "${od}/console.log"
}

run_dcgan_baseline () {
  local seed="$1"
  local od="${OUT}/dcgan_baseline_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u train.py \
    --dataroot "${DC_DATA}" \
    --out "${od}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH_DC}" \
    --zdim 128 \
    --seed "${seed}" \
    | tee "${od}/console.log"
}

run_dcgan_graph () {
  local seed="$1"
  local od="${OUT}/dcgan_graph_diff_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u train.py \
    --dataroot "${DC_DATA}" \
    --out "${od}" \
    --epochs "${EPOCHS}" \
    --batch "${BATCH_DC}" \
    --zdim 128 \
    --seed "${seed}" \
    --graph \
    --graph-backend diff \
    --graph-w 1e-3 \
    --graph-subb 0 \
    --graph-bins 16 --graph-cells 8 --graph-down 4 \
    | tee "${od}/console.log"
}

echo "=== Starting experiments (GPU ${GPU}) ==="
for s in "${SEEDS[@]}"; do
  echo "--- StyleGAN2 baseline (seed ${s}) ---";    run_style2_baseline "${s}"
  echo "--- StyleGAN2 + graph (diff) (seed ${s}) ---"; run_style2_graph "${s}"
  echo "--- DCGAN baseline (seed ${s}) ---";        run_dcgan_baseline "${s}"
  echo "--- DCGAN + graph (diff) (seed ${s}) ---";  run_dcgan_graph "${s}"
done

echo "=== DONE ===
Outputs:
- StyleGAN2 baseline:    ${OUT}/style2_baseline_s<seed>
- StyleGAN2 + graph:     ${OUT}/style2_graph_diff_s<seed>
- DCGAN baseline:        ${OUT}/dcgan_baseline_s<seed>
- DCGAN + graph:         ${OUT}/dcgan_graph_diff_s<seed>
Check: console.log, training_options.json & stats.json (StyleGAN2), sample PNGs; DCGAN: logi, sample gridy."