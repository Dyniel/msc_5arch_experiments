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
from argparse import Namespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendors', 'projected_gan')))

from models.dcgan import Discriminator as DCGAN_Discriminator
from vendors.projected_gan.pg_modules.discriminator import ProjectedDiscriminator

class TestDiscriminatorFactory(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_dcgan_discriminator_creation(self):
        config = {'disc': 'sg2'}
        if config['disc'] == 'pg':
            discriminator = None
        else:
            discriminator = DCGAN_Discriminator().to(self.device)
        self.assertIsInstance(discriminator, DCGAN_Discriminator)

    def test_projected_gan_discriminator_creation(self):
        config = {'disc': 'pg'}
        pg_kwargs = {
            'diffaug': False, 'interp224': True,
            'backbone_kwargs': {'num_discs': 1, 'cond': 0}
        }
        if config['disc'] == 'pg':
            discriminator = ProjectedDiscriminator(**pg_kwargs).to(self.device)
        else:
            discriminator = None
        self.assertIsInstance(discriminator, ProjectedDiscriminator)

    def test_projected_gan_freezes_feature_network(self):
        config = {'disc': 'pg'}
        pg_kwargs = {
            'diffaug': False, 'interp224': True,
            'backbone_kwargs': {'num_discs': 1, 'cond': 0}
        }
        netD = ProjectedDiscriminator(**pg_kwargs).to(self.device)
        netD.train()
        for param in netD.feature_network.parameters():
            param.requires_grad = False
        for param in netD.feature_network.parameters():
            self.assertFalse(param.requires_grad)
        self.assertTrue(any(p.requires_grad for p in netD.discriminator.parameters()))

if __name__ == '__main__':
    unittest.main()
