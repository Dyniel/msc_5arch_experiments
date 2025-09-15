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
import torch
from cleanfid import fid

def main():
    parser = argparse.ArgumentParser(description="Calculate Clean-FID between two directories of images.")
    parser.add_argument('--real-path', type=str, required=True, help='Path to the directory with real images')
    parser.add_argument('--fake-path', type=str, required=True, help='Path to the directory with fake images')
    parser.add_argument('--output-path', type=str, required=True, help='Path to save the output JSON file')
    args = parser.parse_args()
    if not os.path.isdir(args.real_path):
        raise FileNotFoundError(f"Real image directory not found: {args.real_path}")
    if not os.path.isdir(args.fake_path):
        raise FileNotFoundError(f"Fake image directory not found: {args.fake_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Calculating Clean-FID...")
    print(f"Real images: {args.real_path}")
    print(f"Fake images: {args.fake_path}")
    score = fid.compute_fid(
        args.real_path,
        args.fake_path,
        device=device,
        num_workers=4,
    )
    print(f"Clean-FID Score: {score}")
    results = {
        "metric": "clean-fid",
        "score": score,
        "inputs": {
            "real_path": args.real_path,
            "fake_path": args.fake_path,
        },
        "environment": {
            "device": str(device),
            "torch_version": torch.__version__,
        }
    }
    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to: {args.output_path}")

if __name__ == "__main__":
    main()
