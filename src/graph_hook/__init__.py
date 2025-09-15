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
import torch.nn.functional as F
import math

T = 0.008856
P = 7.787

def rgb_to_xyz(rgb):
    rgb = (rgb * 0.5 + 0.5).clamp(0, 1)
    m = torch.tensor([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]], device=rgb.device, dtype=rgb.dtype)
    rgb_linear = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    b, _, h, w = rgb.shape
    rgb_reshaped = rgb_linear.permute(0, 2, 3, 1).reshape(-1, 3)
    xyz = torch.matmul(rgb_reshaped, m.t()).reshape(b, h, w, 3).permute(0, 3, 1, 2)
    xyz[:, 0, :, :] /= 0.95047
    xyz[:, 1, :, :] /= 1.00000
    xyz[:, 2, :, :] /= 1.08883
    return xyz

def xyz_to_lab(xyz):
    f_xyz = torch.where(xyz > T, xyz ** (1/3), P * xyz + (16/116))
    L = 116 * f_xyz[:, 1, :, :] - 16
    a = 500 * (f_xyz[:, 0, :, :] - f_xyz[:, 1, :, :])
    b = 200 * (f_xyz[:, 1, :, :] - f_xyz[:, 2, :, :])
    L_norm = L / 100.0
    return torch.stack([L_norm, a, b], dim=1)

class GraphHook(nn.Module):
    def __init__(self, segments: int, num_iters: int = 5, compactness: float = 10.0, k_neighbors: int = 5):
        super().__init__()
        self.k = segments
        self.m = compactness
        self.iters = num_iters
        self.k_neighbors = k_neighbors

    def loss(self, fake_imgs: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, C, H, W = fake_imgs.shape
        device = fake_imgs.device
        with torch.no_grad():
            lab_imgs = xyz_to_lab(rgb_to_xyz(fake_imgs.detach()))
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=0).float()
        s = int(math.ceil(self.k**0.5))
        grid_y_k, grid_x_k = torch.meshgrid(
            torch.linspace(0, H - 1, s, device=device).long(),
            torch.linspace(0, W - 1, s, device=device).long(),
            indexing='ij'
        )
        flat_y = grid_y_k.flatten()
        flat_x = grid_x_k.flatten()
        if len(flat_y) < self.k:
             raise ValueError(f"Cannot select {self.k} centroids from a grid of size {len(flat_y)}. Lower `k` or increase image resolution.")
        flat_y = flat_y[:self.k]
        flat_x = flat_x[:self.k]
        centroids_xy = torch.stack([flat_y, flat_x], dim=1).float().unsqueeze(0).repeat(B, 1, 1)
        centroids_lab = lab_imgs[:, :, flat_y, flat_x].permute(0, 2, 1)
        norm_xy = H
        norm_lab = 1.0
        pixel_features_lab = lab_imgs.permute(0, 2, 3, 1).reshape(B, H * W, C)
        pixel_features_xy = coords.permute(1, 2, 0).reshape(H * W, 2).unsqueeze(0).repeat(B, 1, 1)
        temperature = 0.1
        for _ in range(self.iters):
            dist_xy = torch.cdist(pixel_features_xy / norm_xy, centroids_xy / norm_xy) ** 2
            dist_lab = torch.cdist(pixel_features_lab / norm_lab, centroids_lab / norm_lab) ** 2
            dist = dist_lab + (self.m / 100.0) * dist_xy
            soft_assign = F.softmax(-dist / temperature, dim=2)
            sum_prob_k = soft_assign.sum(dim=1).unsqueeze(2) + 1e-8
            centroids_xy = torch.bmm(soft_assign.transpose(1, 2), pixel_features_xy) / sum_prob_k
            pixel_features_rgb = fake_imgs.permute(0, 2, 3, 1).reshape(B, H*W, C)
            centroids_rgb = torch.bmm(soft_assign.transpose(1, 2), pixel_features_rgb) / sum_prob_k
            centroids_lab = xyz_to_lab(rgb_to_xyz(centroids_rgb.transpose(1,2).unsqueeze(-1))).squeeze(-1).transpose(1,2)
        centroids_xy_detached = centroids_xy.detach()
        adj_matrix = self._build_knn_graph(centroids_xy_detached, self.k_neighbors)
        num_edges = adj_matrix.sum() / 2
        if num_edges == 0:
            return torch.tensor(0.0, device=device), {}
        c_i = centroids_rgb.unsqueeze(2).expand(-1, -1, self.k, -1)
        c_j = centroids_rgb.unsqueeze(1).expand(-1, self.k, -1, -1)
        diff_sq = ((c_i.float() - c_j.float()) ** 2).sum(dim=3)
        laplacian_loss = (diff_sq * adj_matrix).sum() / (num_edges + 1e-8)
        with torch.no_grad():
            hard_assign = torch.argmax(soft_assign, dim=2)
            label_map = hard_assign.reshape(B, H, W)
            pixel_rgb_sq = (pixel_features_rgb ** 2).sum(dim=2)
            region_mean_sq = (centroids_rgb ** 2).sum(dim=2)
            var_terms = torch.bmm(soft_assign.transpose(1,2), pixel_rgb_sq.unsqueeze(-1)).squeeze(-1)
            region_variance = F.relu(var_terms - region_mean_sq * sum_prob_k.squeeze(-1))
            region_loss = (region_variance.sum(dim=1) / self.k).mean()
            edge_loss = (diff_sq.sqrt() * adj_matrix).sum() / (num_edges + 1e-8)
        stats = {
            'graph/loss_total': laplacian_loss.item(),
            'graph/smooth': laplacian_loss.item(),
            'graph/region': region_loss.item(),
            'graph/edge': edge_loss.item(),
            'label_map': label_map.cpu(),
        }
        return laplacian_loss.to(fake_imgs.dtype), stats

    def _build_knn_graph(self, centroids_xy: torch.Tensor, k: int) -> torch.Tensor:
        B, K, _ = centroids_xy.shape
        dist_matrix = torch.cdist( centroids_xy, centroids_xy)
        dist_matrix.diagonal(dim1=-2, dim2=-1).add_(float('inf'))
        top_k_dists, top_k_indices = torch.topk(dist_matrix, k=k, dim=2, largest=False)
        adj = torch.zeros_like(dist_matrix)
        adj.scatter_(2, top_k_indices, 1)
        adj = (adj + adj.transpose(1, 2)).clamp(0, 1)
        return adj
