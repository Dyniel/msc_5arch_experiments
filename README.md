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

# Fully-Featured GAN Experiment Framework

This project provides a robust, fully-featured, and reproducible framework for training and evaluating Generative Adversarial Networks. Originally a simple DCGAN, it has been augmented with a comprehensive suite of features to enable rigorous experimentation, based on a 20-point checklist for modern ML research.

The core of the project is an experiment comparing a baseline GAN ("OFF") against a GAN augmented with a novel graph-based regularizer ("ON"). The framework is built to support this paired, multi-seed comparison from start to finish.

## Thesis Project Information

This framework was developed as part of the master's thesis: **"Application of Deep Adversarial Networks for Synthesis and Augmentation of Medical Images"** by M.Sc. Eng. Daniel Cieślak, under the supervision of Prof. Jacek Rumiński, D.Sc., Eng. at the Gdańsk University of Technology, Faculty of Electronics, Telecommunications and Informatics.

## Features

This framework implements a wide range of best practices and advanced features:

- **Reproducibility:** Full determinism with seeded workers, plus version-controlled configuration snapshots (`config.json`) including dependencies and git SHA.
- **Performance:** Automatic Mixed Precision (AMP), flexible `num_workers`, and `pin_memory` are supported.
- **Modern Training Techniques:**
  - Exponential Moving Average (EMA) for the generator weights.
  - Two-Timescale Update Rate (TTUR) with separate learning rates for G and D.
  - R1 Gradient Penalty and Differentiable Augmentation (DiffAugment) for discriminator stabilization.
  - `n_critic` support for multiple discriminator updates per generator update.
- **Advanced Graph Regularizer:**
  - A novel "GraphHook" that enforces structural consistency.
  - Highly configurable, with support for ablations (region/edge only, segments, compactness).
  - Performance optimizations like sub-batching with loss scaling and optional caching of real image statistics.
- **Robust Evaluation:**
  - Multi-seed, paired experiment design is orchestrated by the main `run.sh` script.
  - A comprehensive `report.py` script aggregates metrics across all seeds.
  - **Statistical Analysis:** Automatically calculates mean deltas (ON vs. OFF), 95% bootstrap confidence intervals, and paired Wilcoxon p-values.
  - **Anti-Copying Check:** An optional nearest-neighbor LPIPS check to detect training set memorization.
- **Comprehensive Monitoring:**
  - Per-iteration logging of losses, learning rates, VRAM usage, and step timings to both **TensorBoard** and `stats.csv`.
  - Periodic evaluation of FID during training, with checkpointing of the **best model**.
  - **Early stopping** based on a patience for FID improvement.
- **Releasability:** Pinned dependencies, comprehensive README, and automated reporting pipeline.

## Project Structure

```
.
├── models/
│   └── dcgan.py              # Generator and Discriminator architectures
├── src/
│   ├── diff_augment.py       # DiffAugment implementation
│   └── graph_hook_pytorch.py # (Legacy)
├── tools/
│   └── graph_hook.py         # Main GraphHook implementation
├── outputs/                    # Default dir for model checkpoints, logs, samples
├── reports/                    # Default dir for metrics, plots, and reports
├── train.py                  # Main script for training a single model
├── eval_metrics.py           # Script to evaluate a trained model
├── report.py                 # Script to analyze all runs and generate a final report
├── run.sh                    # Orchestrator for the main multi-seed experiment
├── run_ablations.sh          # Demo script for running ablation studies
├── requirements.txt          # Pinned Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## How to Run a Full Experiment

### 1. Installation

Clone the repository and install the pinned dependencies. It is recommended to use a virtual environment.
```bash
git clone <repo_url>
cd <repo_name>
pip install -r requirements.txt
```

### 2. Data Setup

Place your training images in a single folder. For best results, also prepare a separate folder of validation images.

- `/path/to/your/dataset/train/`
- `/path/to/your/dataset/val/`

### 3. Run the Multi-Seed Experiment

The `run.sh` script is the main entry point. It will run the full paired (ON vs. OFF) experiment across 5 different seeds, storing all outputs in the `outputs/` and `reports/` directories.

```bash
# Usage: bash run.sh /path/to/train/data [/path/to/val/data]
bash run.sh /path/to/your/dataset/train /path/to/your/dataset/val
```
*The validation data path is optional but highly recommended for reliable "best model" checkpointing.*

### 4. Generate the Final Report

After `run.sh` completes, the `report.py` script aggregates all the generated metrics, performs statistical analysis, and creates a summary report.

```bash
# Basic report
python report.py

# Report with (slow) LPIPS anti-copying check
python report.py --lpips-check --dataroot /path/to/your/dataset/train
```

This will print the analysis to the console and save the following artifacts in the `reports/` directory:
- `summary.csv`: A CSV file with the full statistical analysis.
- `delta_plot_distance.png`: A plot of the change in FID and KID.
- `delta_plot_quality.png`: A plot of the change in Precision and Recall.
- `comparison_grid_s42.png`: A grid of sample images from the first seed.
- `report.md`: A full markdown report summarizing the experiment.

## Hypothesis Validation

The generated `summary.csv` and `report.md` provide the data needed to validate a hypothesis. For example, to confirm that the GraphHook (`ON`) is better than the baseline (`OFF`), you might look for:
1.  **Statistically Significant FID Improvement:** A negative `Mean Delta` for `clean_fid` with a `p-value` < 0.05.
2.  **Improved Recall without Sacrificing Precision:** A positive `Mean Delta` for `recall` (p < 0.05) while the `Mean Delta` for `precision` is not significantly negative.

## Advanced Usage

### `train.py`

The `train.py` script is highly configurable. See all options with `python train.py --help`. Key arguments include:

- **Data:** `--dataroot`, `--dataroot-val`
- **Resuming:** `--resume /path/to/checkpoint.pt`
- **Performance:** `--amp`, `--workers`
- **Regularization:** `--diffaug-policy 'color,translation'`, `--r1-gamma 10.0`
- **Monitoring:** `--eval-every-iters 2000`, `--early-stop-patience 5`, `--no-tb`
- **GraphHook:** `--graph on`, `--graph-w 0.1`, `--graph-w-warmup-steps 5000`, `--graph-cache-real-stats`, `--graph-region-weight 0.0` (for edge-only ablation), etc.

### `run_ablations.sh`

This script provides examples of how to run single-seed ablation studies to test different components of the GraphHook. See the script for details.
```bash
bash run_ablations.sh /path/to/your/dataset/train
```
