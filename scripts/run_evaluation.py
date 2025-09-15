# =============================================================================
# Copyright (c) Daniel Cieślak
#
# Thesis Title: Application of Deep Adversarial Networks for Synthesis
#               and Augmentation of Medical Images
#
# Author:       M.Sc. Eng. Daniel Cieślak
# Program:      Biomedical Engineering (WETI), M.Sc. full-time, 2023/2024
# Supervisor:   Prof. Jacek Rumiński, D.Sc., Eng.
#
# Notice:
# This script is part of the master's thesis and is legally protected.
# Copying, distribution, or modification without the author's permission is prohibited.
# =============================================================================

import os
import argparse
import subprocess
import json
import logging
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_experiment_name(dir_name):
    try:
        parts = dir_name.split('_')
        seed = int(parts[-1][1:])
        mode = parts[-2]
        model = '_'.join(parts[:-2])
        return model, mode, seed
    except Exception:
        logging.warning(f"Could not parse experiment name: {dir_name}. Skipping.")
        return None, None, None

def run_command(command):
    try:
        logging.info(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed: {' '.join(command)}")
        logging.error(f"Stderr: {e.stderr}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on GAN experiment results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--results_dir", required=True, help="Base directory containing all experiment folders.")
    parser.add_argument("--real_data_path", required=True, help="Path to the real dataset (folder of images for Clean-FID, ZIP for torch-fidelity).")
    parser.add_argument("--output_csv", default="results/summary_raw_metrics.csv", help="Path to save the raw metrics CSV file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to use for metric calculation.")
    parser.add_argument("--fakes_dirname", default="fakes", help="Name of the subdirectory containing generated images.")
    args = parser.parse_args()
    if not os.path.isdir(args.real_data_path):
        logging.warning(f"For Clean-FID, --real_data_path should be a directory. You provided: {args.real_data_path}. Clean-FID may fail if it's not a directory.")
    all_metrics = []
    experiment_dirs = [d for d in os.listdir(args.results_dir) if os.path.isdir(os.path.join(args.results_dir, d))]
    for dir_name in tqdm(experiment_dirs, desc="Evaluating experiments"):
        model, mode, seed = parse_experiment_name(dir_name)
        if not model:
            continue
        fakes_path = os.path.join(args.results_dir, dir_name, args.fakes_dirname)
        if not os.path.isdir(fakes_path):
            logging.warning(f"Fakes directory not found for {dir_name} at {fakes_path}. Skipping.")
            continue
        logging.info(f"--- Processing: {dir_name} ---")
        metrics = {'model': model, 'mode': mode, 'seed': seed}
        fidelity_cmd = [
            'fidelity',
            '--gpu', str(args.gpu_id),
            '--fid', '--kid', '--precision', '--recall',
            '--json',
            '--input1', fakes_path,
            '--input2', args.real_data_path
        ]
        fidelity_output = run_command(fidelity_cmd)
        if fidelity_output:
            try:
                fidelity_metrics = json.loads(fidelity_output)
                metrics['fid'] = fidelity_metrics.get('frechet_inception_distance')
                metrics['kid_mean'] = fidelity_metrics.get('kernel_inception_distance_mean')
                metrics['kid_std'] = fidelity_metrics.get('kernel_inception_distance_std')
                metrics['precision'] = fidelity_metrics.get('precision')
                metrics['recall'] = fidelity_metrics.get('recall')
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON from torch-fidelity for {dir_name}")
        clean_fid_cmd = [
            'clean-fid',
            '--gpu', str(args.gpu_id),
            fakes_path,
            args.real_data_path
        ]
        clean_fid_output = run_command(clean_fid_cmd)
        if clean_fid_output:
            try:
                metrics['clean_fid'] = float(clean_fid_output.strip())
            except ValueError:
                logging.error(f"Failed to parse float from clean-fid for {dir_name}")
        all_metrics.append(metrics)
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        cols = ['model', 'mode', 'seed', 'clean_fid', 'fid', 'kid_mean', 'kid_std', 'precision', 'recall']
        df = df[[c for c in cols if c in df.columns]]
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        logging.info(f"Evaluation results saved to {args.output_csv}")
    else:
        logging.warning("No experiments were evaluated. No CSV file was saved.")

if __name__ == "__main__":
    main()
