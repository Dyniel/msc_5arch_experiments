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

import torch
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage.transform import resize

def _histogram(x, bins, min_val, max_val):
    counts, _ = np.histogram(x, bins=bins, range=(min_val, max_val))
    if counts.sum() > 0:
        counts = counts.astype(np.float32) / counts.sum()
    return counts

def _build_adjacency(segments):
    adj = set()
    h, w = segments.shape
    for y in range(h):
        for x in range(w):
            label = segments[y, x]
            if x > 0 and label != segments[y, x - 1]:
                adj.add(tuple(sorted((label, segments[y, x - 1]))))
            if y > 0 and label != segments[y - 1, x]:
                adj.add(tuple(sorted((label, segments[y - 1, x]))))
    return adj

def get_slic_stats(
    images_gpu: torch.Tensor,
    slic_n_segments: int,
    slic_compactness: float,
    slic_sigma: float,
    slic_max_iter: int,
    slic_resize: int = 0,
    bins: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    with torch.no_grad():
        device = images_gpu.device
        images_np = images_gpu.detach().permute(0, 2, 3, 1).cpu().numpy()
        images_np = (images_np + 1) / 2.0
        batch_size = images_np.shape[0]
        if batch_size == 0:
            return torch.zeros(bins, device=device), torch.zeros(bins, device=device), {}
        all_region_l_means = []
        all_edge_contrasts = []
        total_segments = 0
        images_np_float = img_as_float(images_np)
        for i in range(batch_size):
            img_np = images_np_float[i]
            img_for_slic = img_np
            if slic_resize > 0 and min(img_for_slic.shape[:2]) > slic_resize:
                h_orig, w_orig = img_for_slic.shape[:2]
                if h_orig < w_orig:
                    new_h = slic_resize
                    new_w = int(w_orig * new_h / h_orig)
                else:
                    new_w = slic_resize
                    new_h = int(h_orig * new_w / w_orig)
                img_for_slic = resize(img_np, (new_h, new_w), anti_aliasing=True)
            segments_slic = slic(img_for_slic, n_segments=slic_n_segments, compactness=slic_compactness,
                                 sigma=slic_sigma, max_num_iter=slic_max_iter, start_label=1, channel_axis=-1)
            if slic_resize > 0:
                segments_slic = resize(segments_slic, (img_np.shape[0], img_np.shape[1]),
                                       anti_aliasing=False, order=0, preserve_range=True).astype(np.int64)
            unique_labels = np.unique(segments_slic)
            num_segments = len(unique_labels)
            total_segments += num_segments
            img_lab_np = rgb2lab(img_np)
            l_channel_np = img_lab_np[:, :, 0]
            mean_l_per_segment = np.array([l_channel_np[segments_slic == j].mean() for j in unique_labels if j > 0])
            if mean_l_per_segment.size > 0:
                 all_region_l_means.append(mean_l_per_segment)
            adj = _build_adjacency(segments_slic)
            valid_labels = unique_labels[unique_labels > 0]
            label_to_idx = {label: idx for idx, label in enumerate(valid_labels)}
            edge_contrasts = []
            for u, v in adj:
                if u in label_to_idx and v in label_to_idx:
                    contrast = np.abs(mean_l_per_segment[label_to_idx[u]] - mean_l_per_segment[label_to_idx[v]])
                    edge_contrasts.append(contrast)
            if edge_contrasts:
                all_edge_contrasts.extend(edge_contrasts)
        if all_region_l_means:
            all_region_l_means = np.concatenate(all_region_l_means)
        hist_l_np = _histogram(all_region_l_means, bins, 0, 100)
        hist_edge_np = _histogram(np.array(all_edge_contrasts), bins, 0, 100)
        hist_l = torch.from_numpy(hist_l_np).to(device)
        hist_edge = torch.from_numpy(hist_edge_np).to(device)
        extra_stats = {
            'slic_avg_segments': total_segments / batch_size if batch_size > 0 else 0,
        }
        return hist_l, hist_edge, extra_stats
