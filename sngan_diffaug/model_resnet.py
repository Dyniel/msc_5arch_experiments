# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F

from .spectral_normalization import SpectralNorm
import numpy as np
import math

channels = 3


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
        )
        self.bypass = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.bypass = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, z_dim, image_size=128, gen_size=128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen_size = gen_size
        self.image_size = image_size

        num_blocks = int(math.log2(self.image_size)) - 2  # From 4x4 to image_size

        self.dense = nn.Linear(self.z_dim, 4 * 4 * self.gen_size)

        blocks = []
        in_c = self.gen_size
        for i in range(num_blocks):
            blocks.append(ResBlockGenerator(in_c, self.gen_size, stride=2))

        blocks.extend([
            nn.BatchNorm2d(self.gen_size),
            nn.ReLU(),
            nn.Conv2d(self.gen_size, channels, 3, stride=1, padding=1),
            nn.Tanh()
        ])

        self.model = nn.Sequential(*blocks)

        nn.init.xavier_uniform_(self.dense.weight.data, 1.)

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.gen_size, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, image_size=128, disc_size=128):
        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.disc_size = disc_size

        num_blocks = int(math.log2(
            self.image_size)) - 4  # Downsample to 8x8, e.g. 256 -> 128 (first) -> 64 -> 32 -> 16 -> 8 (4 blocks)

        blocks = [
            FirstResBlockDiscriminator(channels, self.disc_size, stride=2)  # -> image_size/2
        ]

        in_c = self.disc_size
        for i in range(num_blocks):
            blocks.append(ResBlockDiscriminator(in_c, self.disc_size, stride=2))  # -> 4x4

        blocks.extend([
            ResBlockDiscriminator(self.disc_size, self.disc_size, stride=1),
            ResBlockDiscriminator(self.disc_size, self.disc_size, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(8)  # Pool 8x8 to 1x1
        ])

        self.model = nn.Sequential(*blocks)

        self.fc = nn.Linear(self.disc_size, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, self.disc_size))