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
import torch.nn.functional as F

def differentiable_histogram(x: torch.Tensor, bins: int, min_val: float, max_val: float) -> torch.Tensor:
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

def get_diff_stats(
    images_gpu: torch.Tensor,
    down: int = 2,
    cells: int = 16,
    bins: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    n, _, h, w = images_gpu.shape
    device = images_gpu.device
    images_0_1 = (images_gpu + 1) / 2.0
    images_gray = 0.299 * images_0_1[:, 0:1] + 0.587 * images_0_1[:, 1:2] + 0.114 * images_0_1[:, 2:3]
    if down > 1:
        images_gray = F.avg_pool2d(images_gray, kernel_size=down, stride=down)
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(images_gray, sobel_x_kernel, padding='same')
    gy = F.conv2d(images_gray, sobel_y_kernel, padding='same')
    grad_mag = torch.sqrt(gx**2 + gy**2)
    h_c, w_c = images_gray.shape[2], images_gray.shape[3]
    cell_h, cell_w = h_c // cells, w_c // cells
    if cell_h == 0 or cell_w == 0:
        return torch.zeros(bins, device=device), torch.zeros(bins, device=device), {}
    images_gray_cropped = images_gray[:, :, :cells * cell_h, :cells * cell_w]
    grad_mag_cropped = grad_mag[:, :, :cells * cell_h, :cells * cell_w]
    brightness_cells = F.avg_pool2d(images_gray_cropped, kernel_size=(cell_h, cell_w), stride=(cell_h, cell_w))
    grad_mag_cells = F.avg_pool2d(grad_mag_cropped, kernel_size=(cell_h, cell_w), stride=(cell_h, cell_w))
    brightness_vals = brightness_cells.flatten()
    grad_mag_vals = grad_mag_cells.flatten()
    hist_brightness = differentiable_histogram(brightness_vals, bins, 0, 1)
    hist_grad_mag = differentiable_histogram(grad_mag_vals, bins, 0, 4)
    extra_stats = {}
    return hist_brightness, hist_grad_mag, extra_stats
