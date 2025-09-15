# SNGAN-ResNet Backend

This directory contains an implementation of a ResNet-based SNGAN, adapted to function as a training backend within the larger GAN framework.

## Features

*   **Generator**: ResNet-based Generator.
*   **Discriminator**: ResNet-based Discriminator with Spectral Normalization on all convolutional and linear layers.
*   **Loss**: Hinge Loss.
*   **Augmentation**: Differentiable Augmentation (DiffAugment) applied to the discriminator's input.
*   **Regularization**: Optional Graph-based Smoothness Loss applied to the generator.

## Supported Flags

This backend is activated by using `--disc sngan`. It supports the following command-line flags from the main `train.py` script:

### General & Data
*   `--dataroot`: Path to the training dataset.
*   `--out`: Output directory for images, checkpoints, and logs.
*   `--size`: Image size for training (e.g., 128, 256).
*   `--seed`: Manual random seed.
*   `--batch`: Batch size.
*   `--epochs`: Number of training epochs.

### Architecture & Training
*   `--zdim`: Size of the latent vector `z`.
*   `--n-critic`: Number of discriminator updates per generator update.
*   `--g-lr`: Generator learning rate.
*   `--d-lr`: Discriminator learning rate.
*   `--beta1`: Adam optimizer beta1 value.

### Augmentation & Regularization
*   `--diffaug-policy`: Policy for DiffAugment (e.g., `"color,translation,cutout"`).
*   `--graph`: Enable graph loss (`on` or `off`).
*   `--graph-w`: Weight for the graph loss.
*   `--graph-segments`: Number of segments for the graph loss clustering.

### Logging & Snapshots
*   `--snap-iters`: Number of iterations between saving image snapshots and model checkpoints.

## Example Usage

### Standard Training
To run a standard training with SNGAN at 128x128 resolution:
```bash
python train.py --dataroot /path/to/images --out ./output_sngan --disc sngan --size 128 --batch 32 --g-lr 0.0001 --d-lr 0.0002
```

### Training with DiffAugment and Graph Loss
To enable `DiffAugment` and the graph loss regularizer:
```bash
python train.py \
    --dataroot /path/to/images \
    --out ./output_sngan_aug_graph \
    --disc sngan \
    --size 128 \
    --diffaug-policy "color,translation" \
    --graph on \
    --graph-w 0.15 \
    --graph-segments 128
```

## Evaluation

The generated images in the `--out/images/` directory are compatible with the main `eval_metrics.py` script.

## Licensing

*   **This Framework Integration**: The code in `runner.py` and the modifications to `model_resnet.py` are licensed under the main repository's license.
*   **Original SNGAN implementation**: The original code in this directory is based on the [pytorch-spectral-normalization-gan](https://github.com/christiancosgrove/pytorch-spectral-normalization-gan) repository, which is based on the paper "Spectral Normalization for Generative Adversarial Networks" by Miyato et al. (ICLR 2018). The original license is included as `LICENSE`.
*   **DiffAugment**: The `diffaug.py` script is from the paper "Differentiable Augmentation for Data-Efficient GAN Training" by Zhao et al. Its license is available in `THIRD_PARTY/DiffAugment_LICENSE_BSD-2.txt`.
*   **Vendor SNGAN Hash**: The git hash of the vendored code is not available as it was cloned directly.
