#!/usr/bin/env bash
# =================================================================
# SCIENTIFIC COMPARISON SCRIPT
# =================================================================
# This script runs a systematic comparison of all 5 models with
# their available graph regularizer configurations.
#
# It requires two datasets: one for training and one for validation.
#
# By default, it runs 13 experiments. You can disable entire model
# suites by setting the corresponding RUN_XXX variable to false.
# =================================================================
export PYTHONWARNINGS="ignore::UserWarning"
set -euo pipefail

# -----------------------------------------------------------------
# --- CONFIGURATION ---
# -----------------------------------------------------------------
# (Required) Set paths to your training and validation data
TRAIN_DIR=""
VAL_DIR=""

# (Optional) Set output directory and GPU
OUT_BASE_DIR="runs_scientific"
GPU=0
SEED=42

# --- Training Duration ---
# These values are chosen for a ~10-12h total runtime on an A5000.
# Increase them for a more thorough, longer experiment.
KIMG_STYLEGAN=2000   # k-images for StyleGAN2/3 (e.g., 2000 = 2,000,000 images)
EPOCHS_OTHERS=100    # Epochs for DCGAN, PG, SNGAN

# --- Experiment Toggles ---
# Set these to false to skip an entire model suite
RUN_DCGAN=true
RUN_PROJECTED_GAN=true
RUN_SNGAN=true
RUN_STYLEGAN2=true
RUN_STYLEGAN3=true

# --- Sanity Checks ---
if [ -z "$TRAIN_DIR" ] || [ -z "$VAL_DIR" ]; then
    echo "Error: Please set TRAIN_DIR and VAL_DIR at the top of the script."
    exit 1
fi
if [ ! -d "$TRAIN_DIR" ] || [ ! -d "$VAL_DIR" ]; then
    echo "Error: TRAIN_DIR or VAL_DIR is not a valid directory."
    exit 1
fi

# --- Perf Knobs ---
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

# --- Data Preparation ---
# Create resized (256x256) zip archives for StyleGANs
DATA_ZIP_TRAIN="${OUT_BASE_DIR}/train_data_256.zip"
DATA_ZIP_VAL="${OUT_BASE_DIR}/val_data_256.zip"

prepare_zip_dataset() {
    local data_dir="$1"
    local out_zip="$2"
    if [ -f "$out_zip" ]; then
        echo "Zipped dataset already exists: ${out_zip}. Skipping."
        return
    fi
    echo "Creating resized (256x256) zip archive for ${data_dir}..."
    python - <<PY
import os, zipfile, io
try: from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Please run 'pip install Pillow'.")
    exit(1)

data_dir = "${data_dir}"
out_zip = "${out_zip}"
img_size = (256, 256)
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            if not f.lower().endswith(img_extensions): continue
            full_path = os.path.join(root, f)
            try:
                with Image.open(full_path) as img:
                    img_rgb = img.convert('RGB')
                    resized_img = img_rgb.resize(img_size, Image.Resampling.LANCZOS)
                    buffer = io.BytesIO()
                    resized_img.save(buffer, format='PNG')
                    zip_path = os.path.splitext(os.path.relpath(full_path, data_dir))[0] + '.png'
                    zf.writestr(zip_path, buffer.getvalue())
            except Exception as e:
                print(f"Skipping {full_path}: {e}")
print(f"Successfully created {out_zip}")
PY
}

# -----------------------------------------------------------------
# --- UNIVERSAL EVALUATION & REPORTING ---
# -----------------------------------------------------------------

run_evaluation() {
    local exp_name="$1"
    local fakes_dir="$2"
    local metrics_dir="${OUT_BASE_DIR}/metrics"
    mkdir -p "${metrics_dir}"
    local output_json="${metrics_dir}/${exp_name}_fid.json"

    echo "--- [Eval] Evaluating ${exp_name} ---"
    echo "Fake images sourced from: ${fakes_dir}"

    if [ ! -d "${fakes_dir}" ] || [ -z "$(ls -A "${fakes_dir}")" ]; then
        echo "Error: Fake images directory is empty or does not exist: ${fakes_dir}"
        # Create a dummy json with error message
        echo '{ "metric": "clean-fid", "score": null, "error": "Fake directory empty or missing" }' > "${output_json}"
        return 1 # Return with a non-zero exit code to indicate failure
    fi

    # Run the universal evaluation script
    python universal_eval.py \
        --real-path "${VAL_DIR}" \
        --fake-path "${fakes_dir}" \
        --output-path "${output_json}"

    echo "Evaluation for ${exp_name} complete. Results saved to ${output_json}"
}

# -----------------------------------------------------------------
# --- EXPERIMENT DEFINITIONS ---
# -----------------------------------------------------------------

# === DCGAN ===
run_dcgan_base() {
    local exp_name="dcgan-base"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
        --dataroot "${TRAIN_DIR}" --dataroot-val "${VAL_DIR}" --out "${od}" \
        --seed "${SEED}" --disc sg2 --epochs "${EPOCHS_OTHERS}" --batch 16 \
        | tee "${od}/console.log"
    run_evaluation "${exp_name}" "${od}/fakes_ema"
}

# === Projected-GAN ===
run_pg_base() {
    local exp_name="projectedgan-base"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
        --dataroot "${TRAIN_DIR}" --dataroot-val "${VAL_DIR}" --out "${od}" \
        --seed "${SEED}" --disc pg --epochs "${EPOCHS_OTHERS}" --batch 16 \
        | tee "${od}/console.log"
    run_evaluation "${exp_name}" "${od}/fakes_ema"
}
run_pg_graphhook() {
    local exp_name="projectedgan-graphhook"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
        --dataroot "${TRAIN_DIR}" --dataroot-val "${VAL_DIR}" --out "${od}" \
        --seed "${SEED}" --disc pg --epochs "${EPOCHS_OTHERS}" --batch 16 \
        --graph on --graph-w 0.05 \
        | tee "${od}/console.log"
    run_evaluation "${exp_name}" "${od}/fakes_ema"
}

# === SNGAN ===
run_sngan_base() {
    local exp_name="sngan-base"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
        --dataroot "${TRAIN_DIR}" --dataroot-val "${VAL_DIR}" --out "${od}" \
        --seed "${SEED}" --disc sngan --epochs "${EPOCHS_OTHERS}" --batch 16 \
        | tee "${od}/console.log"
    run_evaluation "${exp_name}" "${od}/checkpoints/fakes_final" # SNGAN saves fakes differently
}
run_sngan_graphhook() {
    local exp_name="sngan-graphhook"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u train.py \
        --dataroot "${TRAIN_DIR}" --dataroot-val "${VAL_DIR}" --out "${od}" \
        --seed "${SEED}" --disc sngan --epochs "${EPOCHS_OTHERS}" --batch 16 \
        --graph on --graph-w 0.05 \
        | tee "${od}/console.log"
    run_evaluation "${exp_name}" "${od}/checkpoints/fakes_final"
}

# === StyleGAN2 ===
run_stylegan2_base() {
    local exp_name="stylegan2-base"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style2/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg paper256 --mirror 1 --gamma 10 --aug ada \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --metrics none \
        | tee "${od}/console.log"
    # Evaluation for StyleGAN2 requires generating images first
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style2/generate.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}
run_stylegan2_diff() {
    local exp_name="stylegan2-diff"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style2/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg paper256 --mirror 1 --gamma 10 --aug ada \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --graph --graph-backend diff --graph-w 1e-3 \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style2/generate.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}
run_stylegan2_slic() {
    local exp_name="stylegan2-slic"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style2/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg paper256 --mirror 1 --gamma 10 --aug ada \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --graph --graph-backend slic --graph-w-slic 1e-3 \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style2/generate.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}
run_stylegan2_both() {
    local exp_name="stylegan2-both"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style2/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg paper256 --mirror 1 --gamma 10 --aug ada \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --graph --graph-backend both --graph-w 1e-3 --graph-w-slic 1e-3 \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style2/generate.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}

# === StyleGAN3 ===
run_stylegan3_base() {
    local exp_name="stylegan3-base"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style3/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg stylegan3-t --gamma 10 \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style3/gen_images.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}
run_stylegan3_diff() {
    local exp_name="stylegan3-diff"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style3/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg stylegan3-t --gamma 10 \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --graph --graph-mode diff --graph-w 1e-3 \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style3/gen_images.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}
run_stylegan3_slic() {
    local exp_name="stylegan3-slic"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style3/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg stylegan3-t --gamma 10 \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --graph --graph-mode slic --graph-w 1e-3 \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style3/gen_images.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}
run_stylegan3_both() {
    local exp_name="stylegan3-both"
    local od="${OUT_BASE_DIR}/${exp_name}"
    echo "--- [Train] Starting ${exp_name} ---"
    mkdir -p "${od}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u vendors/style3/train.py \
        --outdir "${od}" --data "${DATA_ZIP_TRAIN}" --gpus 1 \
        --cfg stylegan3-t --gamma 10 \
        --seed "${SEED}" --kimg "${KIMG_STYLEGAN}" \
        --graph --graph-mode both --graph-w-diff 5e-4 --graph-w-slic 5e-4 \
        --metrics none \
        | tee "${od}/console.log"
    local pkl=$(find "${od}" -name "*.pkl" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d' ' -f2-)
    local fakes_dir="${od}/fakes_for_eval"
    CUDA_VISIBLE_DEVICES="${GPU}" python vendors/style3/gen_images.py --outdir="${fakes_dir}" --trunc=1 --seeds=0-4999 --network="${pkl}"
    run_evaluation "${exp_name}" "${fakes_dir}"
}

# -----------------------------------------------------------------
# --- MAIN EXECUTION ---
# -----------------------------------------------------------------

# Prepare datasets
mkdir -p "${OUT_BASE_DIR}"
prepare_zip_dataset "${TRAIN_DIR}" "${DATA_ZIP_TRAIN}"
prepare_zip_dataset "${VAL_DIR}" "${DATA_ZIP_VAL}" # For FID stats later

# Run experiments based on toggles
if $RUN_DCGAN; then
    run_dcgan_base
fi
if $RUN_PROJECTED_GAN; then
    run_pg_base
    run_pg_graphhook
fi
if $RUN_SNGAN; then
    run_sngan_base
    run_sngan_graphhook
fi
if $RUN_STYLEGAN2; then
    run_stylegan2_base
    run_stylegan2_diff
    run_stylegan2_slic
    run_stylegan2_both
fi
if $RUN_STYLEGAN3; then
    run_stylegan3_base
    run_stylegan3_diff
    run_stylegan3_slic
    run_stylegan3_both
fi

echo "=== ALL SCIENTIFIC EXPERIMENTS COMPLETE ==="
echo ""
echo "--- [Report] Generating final summary report ---"
python generate_summary_report.py \
    --results-dir "${OUT_BASE_DIR}/metrics" \
    --output-csv "${OUT_BASE_DIR}/results_summary.csv"

echo "Final report saved to ${OUT_BASE_DIR}/results_summary.csv"
