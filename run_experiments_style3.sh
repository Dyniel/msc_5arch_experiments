#!/usr/bin/env bash
# EXPERIMENTS (≈12h on A5000): StyleGAN3 — baseline & with Graph Regularizer
# Run:  bash run_experiments_style3.sh
export PYTHONWARNINGS="ignore::UserWarning"
set -euo pipefail

# -----------------------------
# CONFIG — EDIT THESE 3 LINES
# -----------------------------
GPU=0
SG3_DATA="/tmp/dset.zip"               # StyleGAN3 dataset (zip or dir)
OUT="runs_experiments_style3"          # Root output folder

# Seeds and training sizes
SEEDS=(42)            # 1 seed for reproducibility
KIMG=1000             # StyleGAN3 training length (thousands of images)
BATCH=32              # Adjust to fit GPU memory

# Perf knobs
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

mkdir -p "${OUT}"

run_style3_baseline () {
  local seed="$1"
  local od="${OUT}/style3_baseline_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style3/train.py \
    --outdir "${od}" \
    --data "${SG3_DATA}" \
    --gpus 1 \
    --cfg stylegan3-t \
    --batch "${BATCH}" \
    --gamma 10 \
    --seed "${seed}" \
    --kimg "${KIMG}" \
    --snap 5 \
    --metrics fid50k_full \
    | tee "${od}/console.log"
}

run_style3_graph_slic () {
  local seed="$1"
  local od="${OUT}/style3_graph_slic_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style3/train.py \
    --outdir "${od}" \
    --data "${SG3_DATA}" \
    --gpus 1 \
    --cfg stylegan3-t \
    --batch "${BATCH}" \
    --gamma 10 \
    --seed "${seed}" \
    --kimg "${KIMG}" \
    --snap 5 \
    --metrics fid50k_full \
    --graph \
    --graph-mode slic \
    --graph-w 1e-3 \
    --graph-segments-slic 128 \
    --graph-compactness 10 \
    --graph-margin 0.05 \
    | tee "${od}/console.log"
}

run_style3_graph_diff () {
  local seed="$1"
  local od="${OUT}/style3_graph_diff_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style3/train.py \
    --outdir "${od}" \
    --data "${SG3_DATA}" \
    --gpus 1 \
    --cfg stylegan3-t \
    --batch "${BATCH}" \
    --gamma 10 \
    --seed "${seed}" \
    --kimg "${KIMG}" \
    --snap 5 \
    --metrics fid50k_full \
    --graph \
    --graph-mode diff \
    --graph-w 1e-3 \
    --graph-segments-diff 128 \
    --graph-tau 0.1 \
    --graph-knn 6 \
    --graph-margin 0.05 \
    | tee "${od}/console.log"
}

run_style3_graph_both () {
  local seed="$1"
  local od="${OUT}/style3_graph_both_s${seed}"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style3/train.py \
    --outdir "${od}" \
    --data "${SG3_DATA}" \
    --gpus 1 \
    --cfg stylegan3-t \
    --batch "${BATCH}" \
    --gamma 10 \
    --seed "${seed}" \
    --kimg "${KIMG}" \
    --snap 5 \
    --metrics fid50k_full \
    --graph \
    --graph-mode both \
    --graph-w-slic 5e-4 \
    --graph-w-diff 5e-4 \
    --graph-segments-slic 128 \
    --graph-compactness 10 \
    --graph-segments-diff 128 \
    --graph-tau 0.1 \
    --graph-knn 6 \
    --graph-margin 0.05 \
    | tee "${od}/console.log"
}

echo "=== Starting StyleGAN3 experiments (GPU ${GPU}) ==="
for s in "${SEEDS[@]}"; do
  echo "--- StyleGAN3 baseline (seed ${s}) ---";      run_style3_baseline "${s}"
  echo "--- StyleGAN3 + graph (slic) (seed ${s}) ---"; run_style3_graph_slic "${s}"
  echo "--- StyleGAN3 + graph (diff) (seed ${s}) ---"; run_style3_graph_diff "${s}"
  echo "--- StyleGAN3 + graph (both) (seed ${s}) ---"; run_style3_graph_both "${s}"
done

echo "=== DONE ===
Outputs:
- StyleGAN3 baseline:    ${OUT}/style3_baseline_s<seed>
- StyleGAN3 + graph slic: ${OUT}/style3_graph_slic_s<seed>
- StyleGAN3 + graph diff: ${OUT}/style3_graph_diff_s<seed>
- StyleGAN3 + graph both: ${OUT}/style3_graph_both_s<seed>
Check: console.log, training_options.json & stats.jsonl, sample PNGs."
