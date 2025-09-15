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
import torch.nn as nn
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.util import img_as_float
from skimage.transform import resize
import torch.nn.functional as F

def differentiable_histogram(x, bins, min_val, max_val):
    if x.numel() == 0:
        return torch.zeros(bins, device=x.device)
    delta = (max_val - min_val) / bins
    x = torch.clamp(x, min_val, max_val - 1e-6)
    bin_centers = torch.linspace(min_val, max_val, bins + 1, device=x.device)[:-1] + delta / 2
    x = x.unsqueeze(-1)
    bin_centers = bin_centers.unsqueeze(0)
    u = (x - bin_centers) / delta
    kernel_vals = torch.relu(1 - torch.abs(u))
    histogram = torch.sum(kernel_vals, dim=0)
    if histogram.sum() > 0:
        histogram = histogram / histogram.sum()
    return histogram

class GraphHook(nn.Module):
    def __init__(self, weight=0.05, n_segments=300, compactness=10, bins=32,
                 region_weight=1.0, edge_weight=1.0, sub_batch_size=4, slic_resize=0):
        super().__init__()
        self.graph_w = weight
        self.n_segments = n_segments
        self.compactness = compactness
        self.bins = bins
        self.region_weight = region_weight
        self.edge_weight = edge_weight
        self.sub_batch_size = sub_batch_size
        self.slic_resize = slic_resize
        self.l1_loss = nn.L1Loss()

    def forward(self, fake_images_gpu, fake_images_np, real_hist_l, real_hist_edge, full_batch_size):
        if self.graph_w <= 0:
            return torch.tensor(0.0, device=fake_images_gpu.device)
        fake_images_norm_gpu = (fake_images_gpu + 1) / 2.0
        hist_fake_l, hist_fake_edge, _, b_sub_fake = self.get_graph_stats(fake_images_norm_gpu, fake_images_np)
        loss_l = self.l1_loss(hist_fake_l, real_hist_l)
        loss_edge = self.l1_loss(hist_fake_edge, real_hist_edge)
        total_loss = self.graph_w * (self.region_weight * loss_l + self.edge_weight * loss_edge)
        if b_sub_fake > 0 and b_sub_fake < full_batch_size:
            scale_factor = full_batch_size / b_sub_fake
            total_loss *= scale_factor
        return total_loss

    def get_graph_stats(self, images_gpu, images_np):
        device = images_gpu.device
        full_batch_size = images_gpu.shape[0]
        effective_sub_batch_size = min(full_batch_size, self.sub_batch_size)
        indices = torch.randperm(full_batch_size, device=device)[:effective_sub_batch_size]
        images_gpu_sub = images_gpu[indices]
        images_np_sub = images_np[indices.cpu().numpy()]
        batch_size, _, h, w = images_gpu_sub.shape
        all_region_l_means = []
        all_edge_contrasts = []
        images_np_float = img_as_float(images_np_sub)
        for i in range(batch_size):
            img_np = images_np_float[i]
            img_for_slic = img_np
            if self.slic_resize > 0:
                img_for_slic = resize(img_np, (self.slic_resize, self.slic_resize), anti_aliasing=True)
            segments_slic = slic(img_for_slic, n_segments=self.n_segments, compactness=self.compactness,
                                 start_label=1, slic_zero=True)
            if self.slic_resize > 0:
                segments_slic = resize(segments_slic, (img_np.shape[0], img_np.shape[1]),
                                       anti_aliasing=False, order=0, preserve_range=True).astype(np.int64)
            img_lab_np = rgb2lab(img_np)
            l_channel_np = img_lab_np[:, :, 0]
            l_channel = torch.from_numpy(l_channel_np).to(device)
            segments = torch.from_numpy(segments_slic).to(device)
            num_segments = len(torch.unique(segments))
            l_means = torch.zeros(num_segments + 1, device=device)
            l_channel_flat = l_channel.flatten()
            segments_flat = segments.flatten()
            l_means.scatter_add_(0, segments_flat, l_channel_flat)
            ones = torch.ones_like(l_channel_flat)
            counts = torch.zeros(num_segments + 1, device=device)
            counts.scatter_add_(0, segments_flat, ones)
            counts[counts == 0] = 1
            mean_l_per_segment = l_means / counts
            mean_l_per_segment = mean_l_per_segment[1:]
            all_region_l_means.append(mean_l_per_segment)
            adj = self._build_adjacency(segments_slic)
            edge_contrasts = []
            for u, v in adj:
                contrast = torch.abs(mean_l_per_segment[u-1] - mean_l_per_segment[v-1])
                edge_contrasts.append(contrast)
            if edge_contrasts:
                all_edge_contrasts.append(torch.stack(edge_contrasts))
        if all_region_l_means:
            all_region_l_means = torch.cat(all_region_l_means)
        else:
            all_region_l_means = torch.tensor([], device=device)
        if all_edge_contrasts:
            all_edge_contrasts = torch.cat(all_edge_contrasts)
        else:
            all_edge_contrasts = torch.tensor([], device=device)
        hist_l = differentiable_histogram(all_region_l_means, self.bins, 0, 100)
        hist_edge = differentiable_histogram(all_edge_contrasts, self.bins, 0, 100)
        return hist_l, hist_edge, full_batch_size, effective_sub_batch_size

    def _build_adjacency(self, segments):
        adj = set()
        horizontal_edges = segments[:, :-1] != segments[:, 1:]
        indices = np.argwhere(horizontal_edges)
        for y, x in indices:
            label1 = segments[y, x]
            label2 = segments[y, x + 1]
            adj.add(tuple(sorted((int(label1), int(label2)))))
        vertical_edges = segments[:-1, :] != segments[1:, :]
        indices = np.argwhere(vertical_edges)
        for y, x in indices:
            label1 = segments[y, x]
            label2 = segments[y + 1, x]
            adj.add(tuple(sorted((int(label1), int(label2)))))
        return adj
