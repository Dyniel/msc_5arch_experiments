#!/bin/bash

# ==============================================================================
# Master Script for Running PyTorch GAN Experiments
#
# This script iterates through a list of models, seeds, and GraphHook modes
# (ON/OFF) to launch a series of training experiments.
#
# USAGE:
#   ./scripts/run_pytorch.sh
#
# PRE-REQUISITES:
# 1. A Conda environment (e.g., 'gan_torch') with all required libraries.
# 2. A prepared dataset in NVLabs ZIP format (created via prepare_data.py).
# 3. The actual training scripts for each model must be available and accept
#    the command-line arguments used below (e.g., --data, --outdir, --seed).
#    You MAY NEED TO MODIFY THE `train_command` variable below to match your
#    specific training scripts.
# ==============================================================================

# --- Configuration ---
export PYTHONUNBUFFERED=1

# An array of your PyTorch models.
# IMPORTANT: These should correspond to how you identify models in your training script.
MODELS=(
    "stylegan2-ada"
    "stylegan3-t"
    "projected-gan"
    "dcgan"
    "lsgan"
    "wgan-gp"
    "dragan"
    "ragan"
    "sngan"
    "sagan"
)

# An array of seeds for reproducibility.
SEEDS=(0 1 2)

# Path to your prepared dataset.
DATA_PATH="data_prepared/dataset_256px.zip"

# Main directory to save all experiment results.
RESULTS_BASE_DIR="results"

# GraphHook weight parameter (only used when GraphHook is ON).
GRAPH_W=0.05

# GPU Management
# This script runs experiments sequentially on one GPU (GPU 0).
# To run in parallel, you would need a more sophisticated job scheduler
# or manually launch this script multiple times with different CUDA_VISIBLE_DEVICES.
# For example:
# CUDA_VISIBLE_DEVICES=0 ./scripts/run_pytorch.sh &
# CUDA_VISIBLE_DEVICES=1 ./scripts/run_pytorch.sh & (after modifying script to run a subset of models)
GPU_ID=0

# --- Main Loop ---
echo "Starting PyTorch GAN experiments..."

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

            # --- Construct the training command ---
            # IMPORTANT: This is a placeholder command. You must adapt it to the
            # command-line API of your actual training script(s).
            # We assume a single `train.py` script that takes a model identifier.
            base_command="python train.py --model ${model} --data ${DATA_PATH} --outdir ${output_dir} --seed ${seed} --gpus ${GPU_ID}"

            # Add GraphHook arguments if mode is 'on'
            if [ "${graph_mode}" = "on" ]; then
                train_command="${base_command} --use_graph_hook=True --graph_w=${GRAPH_W}"
            else
                train_command="${base_command} --use_graph_hook=False"
            fi

            # --- Execute the command ---
            echo "Executing: ${train_command}"

            # Run the command. If it fails, log the error and continue.
            if ! ${train_command}; then
                echo "ERROR: Run ${run_name} failed. Check logs in ${output_dir}."
            fi

            echo "Finished run: ${run_name}"

        done
    done
done

echo "===================================================="
echo "All PyTorch experiments have been launched."
echo "===================================================="
