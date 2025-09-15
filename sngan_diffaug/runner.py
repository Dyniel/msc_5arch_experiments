import os
import json
import random
import time
import torch
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np

# Internal imports
from .model_resnet import Generator, Discriminator
from .diffaug import DiffAugment
from src.graph_hook import GraphHook


def run(config, dataloader, device):
    """
    Main training loop for the SNGAN backend.
    """
    # Set random seed for reproducibility
    if config['seed'] is None:
        config['seed'] = random.randint(1, 10000)
    print(f"SNGAN Runner using Random Seed: {config['seed']}")
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create output directories
    images_dir = os.path.join(config['out'], 'images')
    checkpoints_dir = os.path.join(config['out'], 'checkpoints')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Save config
    with open(os.path.join(config['out'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Initialize models
    netG = Generator(z_dim=config['zdim'], image_size=config['size']).to(device)
    netD = Discriminator(image_size=config['size']).to(device)

    # Setup optimizers - use .get() for safety and correct key names
    optimizerD = optim.Adam(netD.parameters(), lr=config.get('d_lr', 0.0002), betas=(config.get('beta1', 0.5), 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config.get('g_lr', 0.0001), betas=(config.get('beta1', 0.5), 0.999))

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, config['zdim'], device=device)

    # Initialize graph hook if enabled
    graph_hook = None
    if config.get('graph') == 'on':
        print("Graph loss enabled.")
        graph_hook = GraphHook(segments=config['graph_segments'])

    print("Starting SNGAN Training Loop...")
    iters = 0
    training_log_path = os.path.join(config['out'], 'training_log.jsonl')

    for epoch in range(config['epochs']):
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{config['epochs']}")
        for i, (real_imgs, _) in pbar:
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # -----------------
            #  Train Discriminator
            # -----------------
            netD.zero_grad()

            # Train with all-real batch
            real_aug = DiffAugment(real_imgs, policy=config.get('diffaug', ''))
            d_out_real = netD(real_aug)
            loss_d_real = torch.mean(torch.nn.functional.relu(1. - d_out_real))
            loss_d_real.backward()

            # Train with all-fake batch
            noise = torch.randn(b_size, config['zdim'], device=device)
            fake_imgs = netG(noise)

            fake_aug = DiffAugment(fake_imgs.detach(), policy=config.get('diffaug', ''))
            d_out_fake = netD(fake_aug)
            loss_d_fake = torch.mean(torch.nn.functional.relu(1. + d_out_fake))
            loss_d_fake.backward()

            loss_d = loss_d_real + loss_d_fake
            optimizerD.step()

            # -----------------
            #  Train Generator
            # -----------------
            if (i + 1) % config['n_critic'] == 0:
                netG.zero_grad()

                # Generate a new batch of fake images
                gen_noise = torch.randn(b_size, config['zdim'], device=device)
                gen_fake_imgs = netG(gen_noise)

                # Apply augmentations before D
                gen_fake_aug = DiffAugment(gen_fake_imgs, policy=config.get('diffaug', ''))
                d_out_gen = netD(gen_fake_aug)

                # Adversarial loss
                loss_g_adv = -torch.mean(d_out_gen)
                loss_g = loss_g_adv

                loss_g_graph = torch.tensor(0.0)
                if graph_hook:
                    graph_loss_val, _ = graph_hook.loss(gen_fake_imgs)
                    loss_g_graph = config['graph_w'] * graph_loss_val
                    loss_g += loss_g_graph

                loss_g.backward()
                optimizerG.step()

                # --- Logging and Display ---
                pbar.set_postfix({
                    'Loss_D': f'{loss_d.item():.4f}',
                    'Loss_G': f'{loss_g.item():.4f}',
                    'Loss_G_Graph': f'{loss_g_graph.item():.4f}'
                })
                log_entry = {
                    'iter': iters,
                    'epoch': epoch + 1,
                    'loss_D': loss_d.item(),
                    'loss_G': loss_g.item(),
                    'loss_graph': loss_g_graph.item(),
                    'lr_D': optimizerD.param_groups[0]['lr'],
                    'lr_G': optimizerG.param_groups[0]['lr'],
                }
                with open(training_log_path, 'a') as f:
                    f.write(json.dumps(log_entry) + '\n')

            # Save snapshots
            snap_iters = config.get('snap_iters', 1000)
            if (iters > 0 and iters % snap_iters == 0) or (
                    (epoch == config['epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    netG.eval()
                    fake_samples = netG(fixed_noise).detach().cpu()
                    vutils.save_image(fake_samples, os.path.join(images_dir, f'fakes_iter_{iters:06d}.png'),
                                      normalize=True)
                    netG.train()

                torch.save(netG.state_dict(), os.path.join(checkpoints_dir, f'G_{iters}.pt'))
                torch.save(netD.state_dict(), os.path.join(checkpoints_dir, f'D_{iters}.pt'))

            iters += 1

    print("SNGAN Training Finished.")
