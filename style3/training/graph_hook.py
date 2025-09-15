import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage.filters import gaussian


def compute_loss_slic(images, n_segments, compactness, margin):
    """Computes the SLIC-based graph loss with differentiable color/intensity terms."""
    batch_size, _, h, w = images.shape
    device = images.device

    # --- Non-differentiable part: Get SLIC segment labels ---
    with torch.no_grad():
        images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images_np = (images_np + 1) / 2.0

        segments_batch = []
        adj_batch = []
        for i in range(batch_size):
            img_float = img_as_float(images_np[i])
            img_blurred = gaussian(img_float, sigma=0.5, channel_axis=-1)
            segments = slic(
                img_blurred,
                n_segments=n_segments,
                compactness=compactness,
                start_label=0,
                channel_axis=-1,
            )
            segments_batch.append(torch.from_numpy(segments).to(device))

            # Build adjacency graph
            adj = set()
            for y in range(h - 1):
                for x in range(w - 1):
                    label = segments[y, x]
                    right_label = segments[y, x + 1]
                    down_label = segments[y + 1, x]
                    if label != right_label:
                        adj.add(tuple(sorted((label, right_label))))
                    if label != down_label:
                        adj.add(tuple(sorted((label, down_label))))
            adj_batch.append(list(adj))

    # --- Differentiable part: Calculate loss using segment labels ---
    images_gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]  # N x H x W

    total_l_intra = torch.tensor(0.0, device=device)
    total_l_edge = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        segments = segments_batch[i]  # H x W
        adj = adj_batch[i]
        img_gray = images_gray[i]  # H x W

        # Be robust to non-contiguous labels
        num_segments = int(segments.max().item()) + 1

        ones = torch.ones_like(img_gray)
        zeros = torch.zeros(num_segments, device=device, dtype=img_gray.dtype)

        sum_per_segment = zeros.scatter_add(0, segments.view(-1), img_gray.view(-1))
        count_per_segment = zeros.scatter_add(0, segments.view(-1), ones.view(-1))
        mean_per_segment = sum_per_segment / (count_per_segment + 1e-8)

        sum_sq_per_segment = zeros.scatter_add(0, segments.view(-1), img_gray.view(-1).pow(2))
        var_per_segment = (sum_sq_per_segment / (count_per_segment + 1e-8)) - mean_per_segment.pow(2)

        l_intra = var_per_segment.mean()
        total_l_intra = total_l_intra + l_intra

        if len(adj) > 0:
            adj_t = torch.tensor(adj, device=device, dtype=torch.long)
            mu1 = mean_per_segment[adj_t[:, 0]]
            mu2 = mean_per_segment[adj_t[:, 1]]
            l_edge = F.relu(margin - torch.abs(mu1 - mu2)).mean()
            total_l_edge = total_l_edge + l_edge

    l_intra_batch = total_l_intra / batch_size
    l_edge_batch = total_l_edge / batch_size
    total_loss = l_intra_batch + l_edge_batch

    stats = {
        "L_intra_slic": l_intra_batch.item(),
        "L_edge_slic": l_edge_batch.item(),
        "L_graph_slic": total_loss.item(),
    }

    return total_loss, stats


def compute_loss_diff(images, n_segments, tau, knn, margin, compactness=10.0):
    """Computes the differentiable graph-based loss."""
    n, c, h, w = images.shape
    device = images.device

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        ),
        dim=0,
    ).float()
    coords = (coords / (h - 1) * 2 - 1)

    pos_feats = coords * compactness

    pixel_features = torch.cat(
        [images.permute(0, 2, 3, 1), pos_feats.unsqueeze(0).repeat(n, 1, 1, 1).permute(0, 2, 3, 1)],
        dim=-1,
    )
    pixel_features = pixel_features.view(n, h * w, -1)  # N x HW x 5

    indices = torch.randperm(h * w, device=device)[:n_segments]
    centers = pixel_features[:, indices, :].clone()  # N x k x 5

    dists = torch.cdist(pixel_features, centers)  # N x HW x k
    soft_assign = F.softmax(-dists / tau, dim=-1)

    centers_updated = torch.bmm(soft_assign.transpose(1, 2), pixel_features)
    denom = soft_assign.sum(dim=1).unsqueeze(-1) + 1e-8
    centers = centers_updated / denom

    center_rgb = centers[:, :, :3]
    pixel_rgb = pixel_features[:, :, :3]

    dists_updated = torch.cdist(pixel_features, centers)
    soft_assign_updated = F.softmax(-dists_updated / tau, dim=-1)

    dists_rgb = torch.cdist(pixel_rgb, center_rgb)
    l_intra_pixel = (dists_rgb * soft_assign_updated).sum(dim=-1)
    l_intra = l_intra_pixel.mean()

    center_pos = centers[:, :, 3:]
    center_gray = centers[:, :, :3].mean(dim=-1)

    k = centers.shape[1]
    knn_eff = min(knn, max(1, k - 1))
    pos_dists = torch.cdist(center_pos, center_pos)
    _, topk_indices = torch.topk(pos_dists, k=knn_eff + 1, dim=-1, largest=False, sorted=True)
    topk_indices = topk_indices[:, :, 1:]

    if knn_eff > 0:
        center_gray_neighbors = torch.gather(
            center_gray.unsqueeze(1).expand(-1, k, -1),
            2,
            topk_indices,
        )
        l_edge = F.relu(margin - torch.abs(center_gray.unsqueeze(-1) - center_gray_neighbors)).mean()
    else:
        l_edge = torch.zeros((), device=device, dtype=l_intra.dtype)

    total_loss = l_intra + l_edge
    stats = {
        "L_intra_diff": l_intra.item(),
        "L_edge_diff": l_edge.item(),
        "L_graph_diff": total_loss.item(),
    }

    return total_loss, stats


class GraphHook(nn.Module):
    def __init__(
        self,
        graph_mode,
        graph_w,
        graph_w_slic,
        graph_w_diff,
        graph_segments_slic,
        graph_compactness,
        graph_margin,
        graph_every_slic,
        graph_segments_diff,
        graph_tau,
        graph_knn,
        graph_every_diff,
        batch_size,
        **_kwargs,
    ):
        super().__init__()
        self.mode = graph_mode
        self.w = graph_w
        self.w_slic = graph_w_slic
        self.w_diff = graph_w_diff
        self.batch_size = batch_size

        self.segments_slic = graph_segments_slic
        self.compactness = graph_compactness
        self.margin = graph_margin
        self.every_slic = graph_every_slic

        self.segments_diff = graph_segments_diff
        self.tau = graph_tau
        self.knn = graph_knn
        self.every_diff = graph_every_diff

    def __call__(self, gen_img, cur_nimg):
        total_loss = torch.tensor(0.0, device=gen_img.device)
        stats = {}

        # SLIC gating: co kimg
        cur_kimg = cur_nimg / 1000.0
        prev_kimg = (cur_nimg - self.batch_size) / 1000.0

        if self.mode in ["slic", "both"] and (int(cur_kimg / self.every_slic) > int(prev_kimg / self.every_slic)):
            loss_slic, stats_slic = compute_loss_slic(gen_img, self.segments_slic, self.compactness, self.margin)
            weight = self.w_slic if self.mode == "both" else self.w
            total_loss = total_loss + loss_slic * weight
            stats.update(stats_slic)

        # DIFF gating: co krok (minibatch)
        cur_step = cur_nimg // self.batch_size
        prev_step = (cur_nimg - self.batch_size) // self.batch_size
        if self.mode in ["diff", "both"] and (cur_step // self.every_diff) > (prev_step // self.every_diff):
            loss_diff, stats_diff = compute_loss_diff(
                gen_img, self.segments_diff, self.tau, self.knn, self.margin, self.compactness
            )
            weight = self.w_diff if self.mode == "both" else self.w
            total_loss = total_loss + loss_diff * weight
            stats.update(stats_diff)

        if not stats:
            return None, None

        stats["Loss/G/graph_loss"] = total_loss.item()
        return total_loss, stats
