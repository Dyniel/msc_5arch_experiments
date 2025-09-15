# Vendor Information

This document lists the sources of third-party code vendored into this repository.

## Projected-GAN

- **Directory**: `vendors/projected_gan/`
- **Source**: The code was adapted from the official implementation of the paper "Projected GANs Converge Faster".
- **Repository**: [https://github.com/autonomousvision/projected_gan](https://github.com/autonomousvision/projected_gan)
- **Commit**: The specific commit hash is unknown, but the code is based on the state of the repository as of approximately late 2021 / early 2022.
- **Notes**: The original codebase itself builds upon StyleGAN2-ADA, StyleGAN3, FastGAN, and MiDaS. Some modifications were made in this repository to make it compatible with newer library versions (e.g., `timm`).
