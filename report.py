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
import matplotlib as mp
import json
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
import glob
from tqdm import tqdm

try:
    import lpips
except ImportError:
    lpips = None

def aggregate_metrics(reports_dir, outputs_dir, base_name):
    records = []
    for seed_dir in Path(reports_dir).iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith('s'):
            continue
        seed = int(seed_dir.name[1:])
        for metric_file in seed_dir.glob('*_metrics.json'):
            condition = 'on' if '_on_' in metric_file.name else 'off'
            with open(metric_file, 'r') as f:
                data = json.load(f)
            config_path = Path(outputs_dir) / f"{base_name}_s{seed}_{condition}" / "config.json"
            n_gen = 'N/A'
            if config_path.exists():
                with open(config_path, 'r') as f_conf:
                    config_data = json.load(f_conf)
                    n_gen = config_data.get('n_gen', 'N/A')
            data['seed'] = seed
            data['condition'] = condition
            data['n_samples'] = n_gen
            records.append(data)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

def bootstrap_ci(data, n_boot=2000, ci=95):
    boot_means = []
    data = np.asarray(data)
    for _ in range(n_boot):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower_bound = np.percentile(boot_means, (100 - ci) / 2)
    upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(data), lower_bound, upper_bound

def analyze_metrics(df):
    pivot_df = df.pivot(index='seed', columns='condition')
    deltas = pd.DataFrame()
    metrics = sorted(list(set(col[0] for col in pivot_df.columns)))
    for metric in metrics:
        if (metric, 'on') in pivot_df.columns and (metric, 'off') in pivot_df.columns:
            deltas[metric] = pivot_df[(metric, 'on')] - pivot_df[(metric, 'off')]
    print("\n--- Paired Deltas (ON - OFF) ---")
    print(deltas.to_string())
    results = []
    for metric in deltas.columns:
        metric_deltas = deltas[metric].dropna()
        if len(metric_deltas) < 2:
            continue
        mean, ci_low, ci_high = bootstrap_ci(metric_deltas.values)
        try:
            stat, p_value = wilcoxon(metric_deltas.values, zero_method='zsplit')
        except ValueError:
            stat, p_value = np.nan, np.nan
        results.append({
            'Metric': metric,
            'Mean Delta': mean,
            '95% CI': f"[{ci_low:+.3f}, {ci_high:+.3f}]",
            'p-value': p_value,
            'Significant (p<0.05)': '***' if p_value < 0.05 else ''
        })
    return pd.DataFrame(results)

def get_image_paths(dir_path):
    return sorted(list(Path(dir_path).glob('*.png')) + list(Path(dir_path).glob('*.jpg')))

def compute_lpips_scores(real_dir, fake_dir, device):
    if lpips is None:
        print("LPIPS library not found. Skipping LPIPS calculation.")
        return {}
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    real_paths = get_image_paths(real_dir)[:500]
    fake_paths = get_image_paths(fake_dir)
    if not real_paths or not fake_paths:
        print(f"Warning: Could not find images for LPIPS in {real_dir} or {fake_dir}")
        return {}
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    reals = torch.stack([transform(Image.open(p).convert("RGB")) for p in real_paths]).to(device)
    fakes = torch.stack([transform(Image.open(p).convert("RGB")) for p in fake_paths]).to(device)
    min_lpips_scores = []
    print(f"Computing LPIPS distances for {len(fake_paths)} fakes against {len(real_paths)} reals...")
    for i in tqdm(range(fakes.shape[0]), desc="LPIPS"):
        fake_img = fakes[i:i+1]
        dists = loss_fn(fake_img, reals)
        min_lpips_scores.append(dists.min().item())
    return np.array(min_lpips_scores)

def generate_plots(analysis_df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    plot_df = analysis_df.copy()
    plot_df['ci_low_err'] = plot_df['Mean Delta'] - plot_df['ci_low']
    plot_df['ci_high_err'] = plot_df['ci_high'] - plot_df['Mean Delta']
    y_err = plot_df[['ci_low_err', 'ci_high_err']].values.T
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    dist_metrics = plot_df[plot_df['Metric'].isin(['clean_fid', 'fid', 'kid_mean'])]
    colors = ['#d62728' if p < 0.05 else 'gray' for p in dist_metrics['p-value']]
    ax.bar(dist_metrics['Metric'], dist_metrics['Mean Delta'], color=colors, yerr=y_err[:, dist_metrics.index], capsize=5)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_title('Change in Distance Metrics (ON - OFF), Lower is Better', fontsize=16)
    ax.set_ylabel('Mean Delta (95% CI)', fontsize=12)
    fig.tight_layout()
    plt.savefig(out_dir / 'delta_plot_distance.png')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    qual_metrics = plot_df[plot_df['Metric'].isin(['precision', 'recall'])]
    colors = ['#2ca02c' if p < 0.05 else 'gray' for p in qual_metrics['p-value']]
    ax.bar(qual_metrics['Metric'], qual_metrics['Mean Delta'], color=colors, yerr=y_err[:, qual_metrics.index], capsize=5)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_title('Change in Quality Metrics (ON - OFF), Higher is Better', fontsize=16)
    ax.set_ylabel('Mean Delta (95% CI)', fontsize=12)
    fig.tight_layout()
    plt.savefig(out_dir / 'delta_plot_quality.png')
    plt.close(fig)
    print(f"Plots saved to {out_dir}")

def generate_image_grid(outputs_dir, base_name, out_dir, seed):
    out_dir = Path(out_dir)
    off_fakes_pattern = f"{outputs_dir}/{base_name}_s{seed}_off/fakes_epoch_*.png"
    on_fakes_pattern = f"{outputs_dir}/{base_name}_s{seed}_on/fakes_epoch_*.png"
    off_fakes = sorted(glob.glob(off_fakes_pattern))
    on_fakes = sorted(glob.glob(on_fakes_pattern))
    if not off_fakes or not on_fakes:
        print(f"Warning: Could not find epoch snapshots for seed {seed} to generate image grid.")
        return
    img_off = Image.open(off_fakes[-1])
    img_on = Image.open(on_fakes[-1])
    dst = Image.new('RGB', (img_off.width + 50, img_off.height + 50), color='white')
    dst.paste(img_off, (25, 25))
    on_canvas = Image.new('RGB', (img_on.width, img_on.height), color='white')
    on_canvas.paste(img_on, (0, 0))
    dst.paste(on_canvas, (img_off.width + 25, 25))
    grid_path = out_dir / f'comparison_grid_s{seed}.png'
    dst.save(grid_path)
    print(f"Comparison grid saved to {grid_path}")

def generate_markdown_report(analysis_df, lpips_analysis_df, args):
    report_path = Path(args.reports_dir) / 'report.md'
    seed_for_images = args.seed_for_images if args.seed_for_images is not None else pd.read_csv(Path(args.reports_dir) / 'summary.csv')['seed'].unique()[0]
    md = f"# Experiment Report: {args.base_name}\n\n"
    md += "This report summarizes the results of the multi-seed ablation study comparing a baseline GAN (OFF) with a GAN augmented with GraphHook (ON).\n\n"
    md += "## Statistical Analysis of Delta (ON - OFF)\n\n"
    md += "The table below shows the mean difference between the ON and OFF conditions across all seeds. "
    md += "A negative delta is better for distance metrics (FID, KID), while a positive delta is better for quality metrics (Precision, Recall). "
    md += "The 95% confidence interval (CI) and a p-value from a paired Wilcoxon test are provided. "
    md += "Statistically significant results (p < 0.05) are marked.\n\n"
    md += analysis_df.to_markdown(index=False)
    md += "\n\n"
    if lpips_analysis_df is not None:
        md += "## LPIPS Nearest-Neighbor Analysis\n\n"
        md += "This analysis measures the average LPIPS distance to the nearest neighbor in the real dataset for each generated image. Lower scores are better. The 'suspicious rate' is the percentage of images with an LPIPS score below a certain threshold (e.g., 0.1), which could indicate memorization.\n\n"
        md += lpips_analysis_df.to_markdown(index=False)
        md += "\n\n"
    md += "## Plots\n\n"
    md += "### Distance Metrics (Lower is Better)\n"
    md += "![Distance Metrics Plot](delta_plot_distance.png)\n\n"
    md += "### Quality Metrics (Higher is Better)\n"
    md += "![Quality Metrics Plot](delta_plot_quality.png)\n\n"
    md += "## Sample Images\n\n"
    md += f"Below is a comparison of generated image samples from the OFF vs ON conditions for seed {seed_for_images}.\n\n"
    md += f"![Comparison Grid for Seed {seed_for_images}](comparison_grid_s{seed_for_images}.png)\n"
    with open(report_path, 'w') as f:
        f.write(md)
    print(f"Markdown report saved to {report_path}")

def main(args):
    df = aggregate_metrics(args.reports_dir, args.outputs_dir, args.base_name)
    if df.empty:
        print(f"No metric files found in {args.reports_dir}")
        return
    print("--- Aggregated Metrics ---")
    print(df.to_string())
    analysis_results = analyze_metrics(df)
    lpips_full_results = None
    if args.lpips_check:
        all_lpips_scores = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for seed in df['seed'].unique():
            print(f"\n--- Running LPIPS check for seed {seed} ---")
            off_dir = f"{args.outputs_dir}/{args.base_name}_s{seed}_off/fakes_ema/"
            on_dir = f"{args.outputs_dir}/{args.base_name}_s{seed}_on/fakes_ema/"
            scores_off = compute_lpips_scores(args.dataroot, off_dir, device)
            scores_on = compute_lpips_scores(args.dataroot, on_dir, device)
            if len(scores_off) > 0:
                all_lpips_scores.append({'seed': seed, 'condition': 'off', 'lpips_mean': np.mean(scores_off), 'suspicious_rate_0.1': np.mean(scores_off < 0.1)})
            if len(scores_on) > 0:
                all_lpips_scores.append({'seed': seed, 'condition': 'on', 'lpips_mean': np.mean(scores_on), 'suspicious_rate_0.1': np.mean(scores_on < 0.1)})
        if all_lpips_scores:
            lpips_df = pd.DataFrame(all_lpips_scores)
            print("\n--- LPIPS Results (NN) ---")
            print(lpips_df.to_string())
            lpips_full_results = analyze_metrics(lpips_df)
            print("\n--- LPIPS Statistical Analysis ---")
            print(lpips_full_results.to_string())
            analysis_results = pd.concat([analysis_results, lpips_full_results.rename(columns={'Metric': 'Sub-Metric'}).set_index('Sub-Metric')]).reset_index()
    print("\n--- Final Report ---")
    analysis_results[['ci_low', 'ci_high']] = analysis_results['95% CI'].str.strip('[]').str.split(', ', expand=True).astype(float)
    print(analysis_results.to_string())
    summary_path = Path(args.reports_dir) / 'summary.csv'
    analysis_results.to_csv(summary_path, index=False)
    print(f"\nSummary statistics saved to {summary_path}")
    generate_plots(analysis_results, args.reports_dir)
    seed_for_images = args.seed_for_images if args.seed_for_images is not None else df['seed'].unique()[0]
    generate_image_grid(args.outputs_dir, args.base_name, args.reports_dir, seed_for_images)
    generate_markdown_report(analysis_results, lpips_full_results, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate report from multi-seed experiment.")
    parser.add_argument('--reports-dir', type=str, default='reports', help='Directory containing the seed-specific report folders.')
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Directory containing the model output folders (for images).')
    parser.add_argument('--base-name', type=str, default='histopathology_gan', help='Base name for experiment folders.')
    parser.add_argument('--lpips-check', action='store_true', help='Run the nearest-neighbor LPIPS anti-copying check.')
    parser.add_argument('--dataroot', type=str, help='Path to the real dataset (required for LPIPS check).')
    parser.add_argument('--seed-for-images', type=int, default=None, help='Which seed to use for generating the comparison image grid.')
    args = parser.parse_args()
    if args.lpips_check and not args.dataroot:
        parser.error("--dataroot is required for --lpips-check")
    main(args)
