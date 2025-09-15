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

import tensorflow as tf
import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.graph import rag_mean_color

def _normalize_hist(hist):
    if hist.sum() > 0:
        return hist / hist.sum()
    return hist

def _calculate_graph_features_numpy(images_batch, n_segments, compactness, sigma, bins):
    batch_size = images_batch.shape[0]
    batch_region_hists = np.zeros((batch_size, bins), dtype=np.float32)
    batch_edge_hists = np.zeros((batch_size, bins), dtype=np.float32)
    for i in range(batch_size):
        img_numpy = (images_batch[i] * 0.5 + 0.5).clip(0, 1)
        segments = slic(
            img_numpy,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=1,
            channel_axis=-1
        )
        img_lab = rgb2lab(img_numpy)
        rag = rag_mean_color(img_lab, segments)
        region_means_l = np.array([rag.nodes[n]['mean color'][0] for n in rag.nodes])
        region_hist, _ = np.histogram(region_means_l, bins=bins, range=(0, 100))
        batch_region_hists[i] = _normalize_hist(region_hist)
        edge_weights = []
        for u, v, d in rag.edges(data=True):
            color_u = rag.nodes[u]['mean color']
            color_v = rag.nodes[v]['mean color']
            dist = np.linalg.norm(color_u - color_v)
            edge_weights.append(dist)
        if edge_weights:
            edge_weights = np.array(edge_weights)
            edge_hist, _ = np.histogram(edge_weights, bins=bins, range=(0, 100))
            batch_edge_hists[i] = _normalize_hist(edge_hist)
    final_region_hist = batch_region_hists.mean(axis=0).astype(np.float32)
    final_edge_hist = batch_edge_hists.mean(axis=0).astype(np.float32)
    return final_region_hist, final_edge_hist

def graph_hook_loss_tf1(fake_images, real_images, n_segments=300, compactness=10.0, sigma=0, bins=32, region_weight=1.0, edge_weight=1.0):
    def _py_func_wrapper(images):
        return tf.py_function(
            lambda x: _calculate_graph_features_numpy(x, n_segments, compactness, sigma, bins),
            [images],
            [tf.float32, tf.float32]
        )
    real_images_stopped = tf.stop_gradient(real_images)
    real_region_hist, real_edge_hist = _py_func_wrapper(real_images_stopped)
    fake_region_hist, fake_edge_hist = _py_func_wrapper(fake_images)
    real_region_hist.set_shape((bins,))
    real_edge_hist.set_shape((bins,))
    fake_region_hist.set_shape((bins,))
    fake_edge_hist.set_shape((bins,))
    region_loss = tf.losses.absolute_difference(real_region_hist, fake_region_hist)
    edge_loss = tf.losses.absolute_difference(real_edge_hist, fake_edge_hist)
    total_loss = (region_weight * region_loss) + (edge_weight * edge_loss)
    return total_loss
