#!/bin/bash

# ==============================================================================
# Master Script for Running TensorFlow 1.x GAN Experiments
#
# This script iterates through a list of TF1 models, seeds, and GraphHook modes
# (ON/OFF) to launch a series of training experiments inside an NGC container.
#
# USAGE:
#   ./scripts/run_tf1.sh
#
# PRE-REQUISITES:
# 1. NVIDIA Docker (nvidia-container-toolkit) must be installed.
# 2. The NGC TensorFlow 1.15.5 container image must be pulled or available.
# 3. Prepared datasets (ZIP and TFRecords) must be available.
# 4. The actual training scripts for each model must be available and accept
#    the command-line arguments used below. You MAY NEED TO MODIFY THE
#    `train_command` variable to match your specific training scripts.
# ==============================================================================

# --- Configuration ---
export PYTHONUNBUFFERED=1

# An array of your TF1 models.
MODELS=(
    "stylegan-tf1"
    "stylegan2-tf1"
    "pggan"
    "msg-stylegan"
)

# An array of seeds for reproducibility.
SEEDS=(0 1 2)

# Paths to your prepared datasets.
DATA_PATH_ZIP="data_prepared/dataset_256px.zip"
DATA_PATH_TFRECORDS="data_prepared/dataset_256px.tfrecords"

# Main directory to save all experiment results.
RESULTS_BASE_DIR="results"

# GraphHook weight parameter (only used when GraphHook is ON).
GRAPH_W=0.05

# NGC Docker Image
DOCKER_IMAGE="nvcr.io/tensorflow/tensorflow:1.15.5-py3"

# GPU Management (see notes in run_pytorch.sh)
GPU_ID=0

# --- Main Loop ---
echo "Starting TensorFlow 1.x GAN experiments..."

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for graph_mode in "on" "off"; do

            # --- Set up for the current run ---
            run_name="${model}_${graph_mode}_s${seed}"
            output_dir="${RESULTS_BASE_DIR}/${run_name}"

            echo "----------------------------------------------------"
            echo "Starting run: ${run_name}"
            echo "Output directory: ${output_dir}"
            echo "----------------------------------------------------"

            mkdir -p "${output_dir}"

            # --- Select the correct data path ---
            if [ "${model}" = "pggan" ]; then
                data_path_arg="--data ${DATA_PATH_TFRECORDS}"
            else
                data_path_arg="--data ${DATA_PATH_ZIP}"
            fi

            # --- Construct the Python training command ---
            # IMPORTANT: This is a placeholder. Adapt to your TF1 training scripts.
            base_command="python train_tf1.py --model ${model} ${data_path_arg} --outdir ${output_dir} --seed ${seed}"

            if [ "${graph_mode}" = "on" ]; then
                python_command="${base_command} --use_graph_hook=True --graph_w=${GRAPH_W}"
            else
                python_command="${base_command} --use_graph_hook=False"
            fi

            # --- Construct the full Docker command ---
            docker_command="docker run \
                --gpus '\"device=${GPU_ID}\"' \
                --rm \
                -v $(pwd):/workspace \
                -w /workspace \
                ${DOCKER_IMAGE} \
                /bin/bash -c '${python_command}'"

            # --- Execute the command ---
            echo "Executing: ${docker_command}"

            # Using eval to correctly handle the nested quotes in the docker command
            if ! eval ${docker_command}; then
                echo "ERROR: Run ${run_name} failed. Check logs in ${output_dir}."
            fi

            echo "Finished run: ${run_name}"

        done
    done
done

echo "===================================================="
echo "All TensorFlow 1.x experiments have been launched."
echo "===================================================="
