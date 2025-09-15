#!/usr/bin/env bash
# =================================================================
# AUDIT SCRIPT: RUN ALL 5 MODELS
# =================================================================
# This script runs a short, single-seed, demonstrative training
# run for all 5 models found in the repository:
#   1. DCGAN (via train.py --disc sg2)
#   2. Projected-GAN (via train.py --disc pg)
#   3. SNGAN (via train.py --disc sngan)
#   4. StyleGAN2 (via vendors/style2/train.py)
#   5. StyleGAN3 (via vendors/style3/train.py)
#
# Usage:
#   bash run_all.sh /path/to/image/folder
# =================================================================
export PYTHONWARNINGS="ignore::UserWarning"
set -euo pipefail

# -----------------------------
# CONFIG
# -----------------------------
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/image/folder"
    echo "Please provide a path to a directory with training images."
    exit 1
fi

GPU=0
DATA_DIR="$1"                               # A folder of images (e.g. .png, .jpg)
DATA_ZIP="${DATA_DIR}.zip"                  # A zipped version of the folder for StyleGANs
OUT="runs_all_models"                       # Root output folder
SEED=42

# Check if the zip file exists. If not, create it by resizing images to 256x256.
if [ ! -f "$DATA_ZIP" ]; then
    echo "Zipping data for StyleGANs with Python (resizing to 256x256)..."
    python - <<PY
import os, zipfile, io

try:
    from PIL import Image
except ImportError:
    print("Pillow is not installed. Please install it with 'pip install Pillow'")
    exit(1)

data_dir = "${DATA_DIR}"
out_zip = "${DATA_ZIP}"
img_size = (256, 256)
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.lower().endswith(img_extensions):
                continue

            full_path = os.path.join(root, f)
            try:
                with Image.open(full_path) as img:
                    img_rgb = img.convert('RGB')
                    resized_img = img_rgb.resize(img_size, Image.Resampling.LANCZOS)

                    buffer = io.BytesIO()
                    resized_img.save(buffer, format='PNG')
                    buffer.seek(0)

                    rel_path = os.path.relpath(full_path, data_dir)
                    zip_path = os.path.splitext(rel_path)[0] + '.png'
                    zf.writestr(zip_path, buffer.getvalue())
            except Exception as e:
                print(f"Skipping file {full_path} due to error: {e}")

print(f"Created zip with resized images: {out_zip}")
PY
fi

# Short training settings for demonstration
KIMG_SHORT=8
EPOCHS_SHORT=1
BATCH_SIZE=4

# Perf knobs
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

mkdir -p "${OUT}"

# --- Model Runners ---

run_dcgan() {
  local od="${OUT}/1_dcgan_s${SEED}"
  echo "--- [1/5] Running DCGAN ---"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u train.py \
    --dataroot "${DATA_DIR}" \
    --out "${od}" \
    --seed "${SEED}" \
    --disc sg2 \
    --epochs "${EPOCHS_SHORT}" \
    --batch "${BATCH_SIZE}" \
    --eval-every-iters 0 \
    --no-tb \
    | tee "${od}/console.log"
}

run_projected_gan() {
  local od="${OUT}/2_projectedgan_s${SEED}"
  echo "--- [2/5] Running Projected-GAN ---"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u train.py \
    --dataroot "${DATA_DIR}" \
    --out "${od}" \
    --seed "${SEED}" \
    --disc pg \
    --epochs "${EPOCHS_SHORT}" \
    --batch "${BATCH_SIZE}" \
    --eval-every-iters 0 \
    --no-tb \
    | tee "${od}/console.log"
}

run_sngan() {
  local od="${OUT}/3_sngan_s${SEED}"
  echo "--- [3/5] Running SNGAN ---"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u train.py \
    --dataroot "${DATA_DIR}" \
    --out "${od}" \
    --seed "${SEED}" \
    --disc sngan \
    --epochs "${EPOCHS_SHORT}" \
    --batch "${BATCH_SIZE}" \
    --eval-every-iters 0 \
    --no-tb \
    | tee "${od}/console.log"
}

run_stylegan2() {
  local od="${OUT}/4_stylegan2_s${SEED}"
  echo "--- [4/5] Running StyleGAN2 ---"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style2/train.py \
    --outdir "${od}" \
    --data "${DATA_ZIP}" \
    --gpus 1 --cfg paper256 --mirror 1 \
    --aug ada --target 0.6 \
    --gamma 10 --seed "${SEED}" \
    --kimg "${KIMG_SHORT}" \
    --snap 5 \
    --metrics none \
    | tee "${od}/console.log"
}

run_stylegan3() {
  local od="${OUT}/5_stylegan3_s${SEED}"
  echo "--- [5/5] Running StyleGAN3 ---"
  mkdir -p "${od}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
  python -u vendors/style3/train.py \
    --outdir "${od}" \
    --data "${DATA_ZIP}" \
    --gpus 1 \
    --cfg stylegan3-t \
    --batch "${BATCH_SIZE}" \
    --gamma 10 --seed "${SEED}" \
    --kimg "${KIMG_SHORT}" \
    --snap 5 \
    --metrics none \
    | tee "${od}/console.log"
}


# --- Main Execution ---
echo "=== Starting runs for all 5 models (GPU ${GPU}) ==="
run_dcgan
run_projected_gan
run_sngan
run_stylegan2
run_stylegan3

echo "=== ALL 5 DEMO RUNS COMPLETE ==="
echo "Outputs are in the '${OUT}' directory."
