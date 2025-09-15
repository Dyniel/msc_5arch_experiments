import torch
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.util import img_as_float

def differentiable_histogram(x, bins, min_val, max_val):
    """
    Calculates a differentiable histogram of input values.
    This is a soft-assignment histogram using a triangular kernel.
    """
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

def _build_adjacency(segments):
    """
    Builds a 4-neighbor adjacency set from a segmentation map.
    """
    adj = set()
    # Horizontal adjacencies
    horizontal_edges = segments[:, :-1] != segments[:, 1:]
    indices = np.argwhere(horizontal_edges)
    for y, x in indices:
        label1 = segments[y, x]
        label2 = segments[y, x + 1]
        adj.add(tuple(sorted((int(label1), int(label2)))))

    # Vertical adjacencies
    vertical_edges = segments[:-1, :] != segments[1:, :]
    indices = np.argwhere(vertical_edges)
    for y, x in indices:
        label1 = segments[y, x]
        label2 = segments[y + 1, x]
        adj.add(tuple(sorted((int(label1), int(label2)))))

    return adj

def _get_graph_stats(images_gpu, slic_n_segments, slic_compactness, slic_sigma, slic_max_iter, bins=32):
    """
    Computes graph-based statistics (histograms of L-values and edge contrasts) for a batch of images.
    """
    device = images_gpu.device

    images_np = images_gpu.permute(0, 2, 3, 1).cpu().numpy()
    images_np = (images_np + 1) / 2.0

    batch_size, _, _, _ = images_np.shape

    all_region_l_means = []
    all_edge_contrasts = []

    images_np_float = img_as_float(images_np)

    for i in range(batch_size):
        img_np = images_np_float[i]

        segments_slic = slic(img_np, n_segments=slic_n_segments, compactness=slic_compactness,
                             sigma=slic_sigma, max_num_iter=slic_max_iter, start_label=1, slic_zero=True, channel_axis=-1)

        img_lab_np = rgb2lab(img_np)
        l_channel_np = img_lab_np[:, :, 0]

        l_channel = torch.from_numpy(l_channel_np).to(device)
        segments = torch.from_numpy(segments_slic).to(device)

        unique_labels = torch.unique(segments)
        num_segments = len(unique_labels)
        if num_segments <= 1:
            continue

        max_label = unique_labels.max().item()
        l_means = torch.zeros(max_label + 1, device=device)
        l_channel_flat = l_channel.flatten()
        segments_flat = segments.flatten()

        l_means.scatter_add_(0, segments_flat, l_channel_flat)

        ones = torch.ones_like(l_channel_flat)
        counts = torch.zeros(max_label + 1, device=device)
        counts.scatter_add_(0, segments_flat, ones)
        counts[counts == 0] = 1

        mean_l_per_segment = l_means / counts

        # Filter out zero-label (if any) and use only valid segment means
        valid_labels = unique_labels[unique_labels > 0]
        mean_l_per_segment_filtered = mean_l_per_segment[valid_labels]
        all_region_l_means.append(mean_l_per_segment_filtered)

        adj = _build_adjacency(segments_slic)

        edge_contrasts = []
        label_to_idx = {label.item(): idx for idx, label in enumerate(valid_labels)}

        for u, v in adj:
            if u in label_to_idx and v in label_to_idx:
                 contrast = torch.abs(mean_l_per_segment_filtered[label_to_idx[u]] - mean_l_per_segment_filtered[label_to_idx[v]])
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

    hist_l = differentiable_histogram(all_region_l_means, bins, 0, 100)
    hist_edge = differentiable_histogram(all_edge_contrasts, bins, 0, 100)

    return hist_l, hist_edge

def graph_loss(images: torch.Tensor, real_images: torch.Tensor, *, slic_n_segments: int, slic_compactness: float, slic_sigma: float, slic_max_iter: int) -> torch.Tensor:
    """
    Computes the graph-based regularizer loss by comparing statistics of fake and real images.
    """
    if slic_n_segments <= 0:
        return torch.tensor(0.0, device=images.device)

    fake_hist_l, fake_hist_edge = _get_graph_stats(images, slic_n_segments, slic_compactness, slic_sigma, slic_max_iter)

    with torch.no_grad():
        real_hist_l, real_hist_edge = _get_graph_stats(real_images, slic_n_segments, slic_compactness, slic_sigma, slic_max_iter)

    loss_l = torch.nn.functional.l1_loss(fake_hist_l, real_hist_l)
    loss_edge = torch.nn.functional.l1_loss(fake_hist_edge, real_hist_edge)

    return loss_l + loss_edge
