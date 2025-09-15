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

import unittest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_hook import GraphHook

class TestGraphHook(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hook = GraphHook(segments=16, k_neighbors=2).to(self.device)
        self.batch_size = 2
        self.img_size = 32
        self.fake_imgs = torch.randn(self.batch_size, 3, self.img_size, self.img_size,
                                     device=self.device, requires_grad=True)

    def test_loss_output_shape_and_type(self):
        loss, stats = self.hook.loss(self.fake_imgs)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(loss.requires_grad)

    def test_backward_pass(self):
        loss, _ = self.hook.loss(self.fake_imgs)
        loss.backward()
        self.assertIsNotNone(self.fake_imgs.grad)
        self.assertGreater(self.fake_imgs.grad.abs().sum(), 0)

    def test_determinism(self):
        torch.manual_seed(42)
        loss1, stats1 = self.hook.loss(self.fake_imgs)
        torch.manual_seed(42)
        fake_imgs_2 = self.fake_imgs.detach().clone().requires_grad_(True)
        loss2, stats2 = self.hook.loss(fake_imgs_2)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)
        self.assertTrue(torch.allclose(stats1['label_map'], stats2['label_map']))

    def test_stats_dictionary(self):
        _, stats = self.hook.loss(self.fake_imgs)
        self.assertIsInstance(stats, dict)
        expected_keys = ['graph/loss_total', 'graph/smooth', 'graph/region', 'graph/edge', 'label_map']
        for key in expected_keys:
            self.assertIn(key, stats)
        self.assertEqual(stats['label_map'].shape, (self.batch_size, self.img_size, self.img_size))

if __name__ == '__main__':
    unittest.main()
