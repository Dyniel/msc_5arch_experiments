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

import argparse
import json
import os
import glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Generate a summary CSV from individual metric JSON files.")
    parser.add_argument('--results-dir', type=str, required=True, help='Directory containing the JSON result files')
    parser.add_argument('--output-csv', type=str, required=True, help='Path to save the output summary CSV file')
    args = parser.parse_args()
    if not os.path.isdir(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
    json_files = glob.glob(os.path.join(args.results_dir, '*.json'))
    if not json_files:
        print(f"No JSON files found in {args.results_dir}. No report generated.")
        return
    results_list = []
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            filename = os.path.basename(json_path)
            exp_name = filename.replace('_fid.json', '')
            results_list.append({
                'experiment_name': exp_name,
                'fid_score': data.get('score'),
                'error': data.get('error', None)
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not parse {json_path}. Error: {e}")
            filename = os.path.basename(json_path)
            exp_name = filename.replace('_fid.json', '')
            results_list.append({
                'experiment_name': exp_name,
                'fid_score': None,
                'error': f"Failed to parse JSON: {e}"
            })
    if not results_list:
        print("No valid results found. No report generated.")
        return
    df = pd.DataFrame(results_list)
    df = df.sort_values(by='experiment_name').reset_index(drop=True)
    df = df[['experiment_name', 'fid_score', 'error']]
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Summary report generated successfully: {args.output_csv}")
    print("\n--- Results Summary ---")
    print(df.to_string())

if __name__ == "__main__":
    main()
