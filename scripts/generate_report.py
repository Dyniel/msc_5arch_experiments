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
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bootstrap_ci(data, n_bootstraps=1000, alpha=0.05):
    if len(data.dropna()) < 2:
        return np.nan, np.nan, np.nan
    means = np.zeros(n_bootstraps)
    data_array = np.array(data)
    for i in range(n_bootstraps):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        means[i] = np.mean(sample)
    lower_bound = np.percentile(means, (alpha / 2) * 100)
    upper_bound = np.percentile(means, (1 - alpha / 2) * 100)
    mean_val = np.mean(data)
    return mean_val, lower_bound, upper_bound

def main():
    parser = argparse.ArgumentParser(
        description="Generate aggregated results and a PDF report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_csv", required=True, help="Path to the raw metrics CSV file from run_evaluation.py.")
    parser.add_argument("--output_dir", default="reports", help="Directory to save the aggregated CSV and PDF report.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Loading raw data from {args.input_csv}")
    try:
        df_raw = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        logging.error(f"Input file not found: {args.input_csv}. Exiting.")
        return
    metrics_to_agg = ['clean_fid', 'fid', 'kid_mean', 'precision', 'recall']
    aggregated_results = []
    for (model, mode), group in df_raw.groupby(['model', 'mode']):
        result_row = {'model': model, 'mode': mode}
        for metric in metrics_to_agg:
            if metric in group:
                mean, ci_low, ci_high = bootstrap_ci(group[metric])
                result_row[f'{metric}_mean'] = mean
                result_row[f'{metric}_ci_low'] = ci_low
                result_row[f'{metric}_ci_high'] = ci_high
        aggregated_results.append(result_row)
    df_agg = pd.DataFrame(aggregated_results)
    agg_csv_path = os.path.join(args.output_dir, "summary_aggregated.csv")
    df_agg.to_csv(agg_csv_path, index=False)
    logging.info(f"Aggregated results saved to {agg_csv_path}")
    pdf_path = os.path.join(args.output_dir, "report.pdf")
    logging.info(f"Generating PDF report at {pdf_path}...")
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(0.5, 0.6, 'GAN Evaluation Report', ha='center', va='center', fontsize=24)
        fig.text(0.5, 0.5, f'Source: {os.path.basename(args.input_csv)}', ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.45, f'Date: {pd.to_datetime("today").date()}', ha='center', va='center', fontsize=16)
        pdf.savefig(fig)
        plt.close()
        df_pivot = df_agg.set_index(['model', 'mode'])
        for metric in metrics_to_agg:
            if f'{metric}_mean' not in df_pivot.columns:
                continue
            fig = plt.figure(figsize=(11.7, 8.3))
            ax = fig.add_subplot(111)
            ax.set_title(f'Aggregated Results: {metric.upper()}', fontsize=16, pad=20)
            table_data = df_pivot.reset_index()
            table_data['value'] = table_data.apply(
                lambda r: f"{r[f'{metric}_mean']:.3f} [{r[f'{metric}_ci_low']:.3f}, {r[f'{metric}_ci_high']:.3f}]"
                if pd.notna(r[f'{metric}_mean']) else "N/A", axis=1
            )
            table_pivot = table_data.pivot(index='model', columns='mode', values='value')
            ax.axis('off')
            table = ax.table(cellText=table_pivot.values, colLabels=table_pivot.columns, rowLabels=table_pivot.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            pdf.savefig(fig)
            plt.close()
            fig = plt.figure(figsize=(11.7, 8.3))
            ax = fig.add_subplot(111)
            sns.barplot(data=df_agg, x='model', y=f'{metric}_mean', hue='mode', ax=ax)
            df_agg['err'] = df_agg[f'{metric}_ci_high'] - df_agg[f'{metric}_ci_low']
            for i, p in enumerate(ax.patches):
                group_size = len(df_agg['model'].unique())
                model_name = df_agg['model'].unique()[i % group_size]
                mode_name = df_agg['mode'].unique()[i // group_size]
                err_val = df_agg[(df_agg['model'] == model_name) & (df_agg['mode'] == mode_name)]['err'].values[0] / 2
                ax.errorbar(p.get_x() + p.get_width() / 2., p.get_height(), yerr=err_val, color='black', capsize=4)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Comparison Plot: {metric.upper()}', fontsize=16)
            plt.ylabel(metric.upper())
            plt.xlabel('Model')
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    logging.info("PDF report generated successfully.")

if __name__ == "__main__":
    main()
