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

import time
import torch
import torch.nn.functional as F
from . import backend_slic
from . import backend_diff

@torch.amp.autocast('cuda', enabled=False)
def compute_graph_loss(
    fake_img: torch.Tensor,
    real_img: torch.Tensor,
    backend: str,
    sub_batch: int | None,
    diff_kwargs: dict,
    slic_kwargs: dict,
) -> tuple[torch.Tensor, float, dict]:
    t0 = time.time()
    fake_img = fake_img.float()
    real_img = real_img.float()
    loss = torch.tensor(0.0, device=fake_img.device)
    extra_stats = {}
    if backend == 'diff':
        real_hist_l, real_hist_edge, slic_stats = backend_slic.get_slic_stats(real_img, **slic_kwargs)
        extra_stats.update(slic_stats)
        if sub_batch is not None and sub_batch > 0 and fake_img.size(0) > sub_batch:
            fake_chunks = torch.split(fake_img, sub_batch)
            losses = []
            for chunk in fake_chunks:
                fake_hist_bright, fake_hist_grad, _ = backend_diff.get_diff_stats(chunk, **diff_kwargs)
                loss_bright = F.l1_loss(fake_hist_bright, real_hist_l)
                loss_grad = F.l1_loss(fake_hist_grad, real_hist_edge)
                losses.append(loss_bright + loss_grad)
            loss = torch.stack(losses).mean()
        else:
            fake_hist_bright, fake_hist_grad, _ = backend_diff.get_diff_stats(fake_img, **diff_kwargs)
            loss_bright = F.l1_loss(fake_hist_bright, real_hist_l)
            loss_grad = F.l1_loss(fake_hist_grad, real_hist_edge)
            loss = loss_bright + loss_grad
        extra_stats['l1_loss'] = loss.item()
    elif backend == 'slic':
        real_hist_l, real_hist_edge, _ = backend_slic.get_slic_stats(real_img, **slic_kwargs)
        if sub_batch is not None and sub_batch > 0 and fake_img.size(0) > sub_batch:
            fake_chunks = torch.split(fake_img.detach(), sub_batch)
            losses = []
            for chunk in fake_chunks:
                fake_hist_l, fake_hist_edge, _ = backend_slic.get_slic_stats(chunk, **slic_kwargs)
                loss_l = F.l1_loss(fake_hist_l, real_hist_l)
                loss_edge = F.l1_loss(fake_hist_edge, real_hist_edge)
                losses.append(loss_l + loss_edge)
            loss = torch.stack(losses).mean()
        else:
            fake_hist_l, fake_hist_edge, _ = backend_slic.get_slic_stats(fake_img.detach(), **slic_kwargs)
            loss_l = F.l1_loss(fake_hist_l, real_hist_l)
            loss_edge = F.l1_loss(fake_hist_edge, real_hist_edge)
            loss = loss_l + loss_edge
        extra_stats['l1_loss'] = loss.item()
    else:
        raise ValueError(f"Unknown graph backend: '{backend}'")
    ms = (time.time() - t0) * 1000.0
    return loss, ms, extra_stats
