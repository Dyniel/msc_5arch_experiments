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

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
from PIL import Image
import json
import subprocess
import copy
import sys
import time
import shutil
from pathlib import Path
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from cleanfid import fid
from cleanfid.features import build_feature_extractor
import scipy.linalg
import torch.nn.functional as F

from models.dcgan import Generator as DCGAN_G, Discriminator as DCGAN_D, weights_init
from src.diff_augment import DiffAugment
from torch.utils.data import Dataset
from src.graph_hook import GraphHook as NewGraphHook

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class Logger:
    def __init__(self, out_dir, use_tb=True):
        self.out_dir = Path(out_dir)
        self.use_tb = use_tb
        if self.use_tb:
            self.writer = SummaryWriter(self.out_dir)
        self.train_log_path = self.out_dir / 'train_log.csv'
        self.metrics_log_path = self.out_dir / 'metrics.csv'
        if not self.train_log_path.exists():
            with open(self.train_log_path, 'w') as f:
                f.write(
                    'step,epoch,iter,loss_d,loss_d_r1,loss_g,loss_g_bce,'
                    'loss_g_graph_diff,loss_g_graph_slic,graph_l1_loss,'
                    'graph/loss_total,graph/smooth,graph/region,graph/edge,'
                    'lr_g,lr_d,time_data,time_graph_ms_diff,time_graph_ms_slic,time_step,'
                    'vram_alloc_gb,vram_max_gb,pg/n_trainable_params,pg/params_frozen\n'
                )
        if not self.metrics_log_path.exists():
            with open(self.metrics_log_path, 'w') as f:
                f.write('step,epoch,fid_val\n')

    def log(self, data, step):
        if self.use_tb:
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, step)
        data['step'] = step
        if 'eval/fid' in data:
            with open(self.metrics_log_path, 'a') as f:
                f.write(f"{step},{data.get('epoch', -1)},{data['eval/fid']:.4f}\n")
        else:
            with open(self.train_log_path, 'a') as f:
                f.write(f"{step},{data.get('epoch', -1)},{data.get('iter', -1)},"
                        f"{data.get('loss/d', 0):.4f},{data.get('loss/d_r1', 0):.4f},"
                        f"{data.get('loss/g', 0):.4f},{data.get('loss/g_bce', 0):.4f},"
                        f"{data.get('loss/g_graph_diff', 0):.4f},{data.get('loss/g_graph_slic', 0):.4f},{data.get('graph/l1_loss', 0):.4f},"
                        f"{data.get('graph/loss_total', 0):.4f},{data.get('graph/smooth', 0):.4f},{data.get('graph/region', 0):.4f},{data.get('graph/edge', 0):.4f},"
                        f"{data.get('lr/g', 0):.6f},{data.get('lr/d', 0):.6f},"
                        f"{data.get('time/data_ms', 0):.4f},{data.get('time/graph_ms_diff', 0):.4f},{data.get('time/graph_ms_slic', 0):.4f},{data.get('time/step_ms', 0):.4f},"
                        f"{data.get('vram/alloc_gb', 0):.2f},{data.get('vram/max_gb', 0):.2f},"
                        f"{data.get('pg/n_trainable_params', 0)},{data.get('pg/params_frozen', 0)}\n")

    def close(self):
        if self.use_tb:
            self.writer.close()

def make_reference_stats(name, data_path, fid_model, num_workers=4, device=torch.device("cuda")):
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "clean-fid")
    stats_path = os.path.join(cache_dir, f"{name}.npz")
    if os.path.exists(stats_path):
        print(f"Reference FID stats '{name}' already exist at {stats_path}. Skipping generation.")
        return
    print(f"Generating reference FID stats '{name}' from {data_path}...")
    fid.make_custom_stats(name, fdir=data_path, model=fid_model, num_workers=num_workers, device=device)
    print(f"Successfully created reference FID stats '{name}'.")

def frechet_distance_stable(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariances have different dimensions"
    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    sigma1_eps = sigma1 + offset
    sigma2_eps = sigma2 + offset
    covmean, _ = scipy.linalg.sqrtm(sigma1_eps.dot(sigma2_eps), disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: FID calculation produced significant imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

def compute_fid_stable(fakes_dir, dataset_name_or_path, config, device, fid_model):
    features_fake = fid.get_folder_features(
        fakes_dir, model=fid_model, num_workers=config['workers'], device=device, mode="clean"
    )
    mu_fake = np.mean(features_fake, axis=0)
    sigma_fake = np.cov(features_fake, rowvar=False)
    if os.path.isdir(dataset_name_or_path):
        print(f"Calculating reference stats from directory: {dataset_name_or_path}")
        features_ref = fid.get_folder_features(
            dataset_name_or_path, model=fid_model, num_workers=config['workers'], device=device, mode="clean"
        )
        mu_ref = np.mean(features_ref, axis=0)
        sigma_ref = np.cov(features_ref, rowvar=False)
    else:
        print(f"Loading reference stats: {dataset_name_or_path}")
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "clean-fid")
        stats_path = os.path.join(cache_dir, f"{dataset_name_or_path}.npz")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"Pre-calculated stats not found at {stats_path}. Please run with a valid directory first to generate them.")
        data = np.load(stats_path)
        mu_ref, sigma_ref = data['mu'], data['sigma']
    return frechet_distance_stable(mu_fake, sigma_fake, mu_ref, sigma_ref)

def run_periodic_eval(netG_ema, config, device, iters, val_path, fid_model):
    eval_dir = Path(config['out']) / f"eval_{iters}"
    fakes_dir = eval_dir / "fakes"
    os.makedirs(fakes_dir, exist_ok=True)
    netG_ema.eval()
    n_fakes = config.get('eval_n_fakes', 1024)
    print(f"\nGenerating {n_fakes} fakes for evaluation at step {iters}...")
    with torch.no_grad():
        for i in tqdm(range(n_fakes), desc="Generating fakes for FID"):
            noise = torch.randn(1, config['zdim'], 1, 1, device=device)
            with autocast(enabled=config['amp']):
                fake_img = netG_ema(noise)
            vutils.save_image(fake_img, fakes_dir / f'fake_{i:04d}.png', normalize=True)
    print("Calculating Clean-FID (stable)...")
    fid_ref = config.get('ref_name') or val_path
    fid_score = compute_fid_stable(str(fakes_dir), fid_ref, config, device, fid_model)
    print(f"FID at step {iters}: {fid_score:.4f}")
    shutil.rmtree(eval_dir)
    netG_ema.train()
    return fid_score

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=IMG_EXTENSIONS):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for root, _, fnames in sorted(os.walk(self.root_dir)):
            for fname in sorted(fnames):
                if fname.lower().endswith(extensions):
                    path = os.path.join(root, fname)
                    self.image_paths.append(path)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"Found no images with extensions {extensions} in {self.root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_system_info():
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "N/A"
    try:
        pip_freeze = subprocess.check_output(['pip', 'freeze']).decode('utf-8').strip().split('\n')
    except (subprocess.CalledProcessError, FileNotFoundError):
        pip_freeze = "N/A"
    cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
    return {"git_sha": git_hash, "dependencies": pip_freeze, "cuda_version": cuda_version}

def main(config):
    os.makedirs(config['out'], exist_ok=True)
    if config['seed'] is None:
        config['seed'] = random.randint(1, 10000)
    print(f"Random Seed: {config['seed']}")
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    g = torch.Generator()
    g.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if not config['resume']:
        system_info = get_system_info()
        config.update(system_info)
        with open(os.path.join(config['out'], 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
    logger = Logger(config['out'], use_tb=not config['no_tb'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Building InceptionV3 feature extractor for FID...")
    fid_model = build_feature_extractor("clean", device)
    dataset = ImageDataset(root_dir=config['dataroot'],
                           transform=transforms.Compose([
                               transforms.Resize(config['size']),
                               transforms.CenterCrop(config['size']),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch'],
                                             shuffle=True, num_workers=config['workers'],
                                             worker_init_fn=seed_worker, generator=g,
                                             pin_memory=True)
    val_path = config.get('dataroot_val')
    if config['eval_every_iters'] > 0:
        if not val_path:
            raise ValueError("--dataroot-val is required for periodic evaluation (when --eval-every-iters > 0).")
        if not os.path.isdir(val_path):
            raise FileNotFoundError(f"Validation data path for periodic evaluation not found: {val_path}")
    if config['ref_name']:
        stats_source_path = val_path if val_path else config['dataroot']
        if not os.path.isdir(stats_source_path):
            raise FileNotFoundError(f"Data path for generating reference stats '{config['ref_name']}' not found: {stats_source_path}")
        make_reference_stats(config['ref_name'], stats_source_path, fid_model, num_workers=config['workers'], device=device)
    pg_params_frozen = 0
    pg_n_trainable_params = 0
    if config['disc'] == 'pg':
        print("Using Projected-GAN Discriminator")
        pg_vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vendors', 'projected_gan'))
        if pg_vendor_path not in sys.path:
            sys.path.insert(0, pg_vendor_path)
        from vendors.projected_gan.pg_modules.discriminator import ProjectedDiscriminator
        pg_kwargs = {
            'diffaug': False,
            'interp224': True,
            'backbone_kwargs': {
                'num_discs': 4,
                'cond': 0,
            }
        }
        netG = DCGAN_G(z_dim=config['zdim']).to(device)
        netD = ProjectedDiscriminator(**pg_kwargs).to(device)
        netG.apply(weights_init)
        netD.train()
        for param in netD.feature_network.parameters():
            param.requires_grad = False
        feature_params = list(netD.feature_network.parameters())
        disc_params = list(netD.discriminator.parameters())
        pg_params_frozen = sum(p.numel() for p in feature_params)
        pg_n_trainable_params = sum(p.numel() for p in disc_params)
        print(
            f"  -> Projected-GAN: {pg_n_trainable_params / 1e6:.2f}M trainable params, {pg_params_frozen / 1e6:.2f}M frozen params.")
        optimizerD_params = netD.discriminator.parameters()
    elif config['disc'] == 'sngan':
        print("Using SNGAN-ResNet Backend via Runner")
        from vendors.sngan_diffaug import runner as sngan_runner
        sngan_runner.run(config, dataloader, device)
        print("SNGAN runner finished.")
        return
    else:
        print("Using DCGAN Discriminator")
        netG = DCGAN_G(z_dim=config['zdim']).to(device)
        netD = DCGAN_D().to(device)
        netG.apply(weights_init)
        netD.apply(weights_init)
        optimizerD_params = netD.parameters()
    netG_ema = copy.deepcopy(netG).to(device)
    netG_ema.eval()
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        print(f"Using {torch.cuda.device_count()} GPUs!")
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        netG_ema = nn.DataParallel(netG_ema)
    criterion = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(optimizerD_params, lr=config['lr_d'], betas=(config['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=config['lr_g'], betas=(config['beta1'], 0.999))
    scaler = GradScaler(enabled=config['amp'])
    fixed_noise = torch.randn(64, config['zdim'], 1, 1, device=device)
    real_label = 1.
    fake_label = 0.
    start_epoch = 0
    iters = 0
    best_fid = float('inf')
    if config['resume']:
        print(f"Resuming training from checkpoint: {config['resume']}")
        checkpoint = torch.load(config['resume'], map_location=device)
        netG.load_state_dict(checkpoint['generator_state_dict'])
        netD.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'netG_ema_state_dict' in checkpoint:
            netG_ema.load_state_dict(checkpoint['netG_ema_state_dict'])
        else:
            netG_ema = copy.deepcopy(netG)
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        if 'scaler_state_dict' in checkpoint and config['amp']:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        iters = checkpoint.get('iters', 0)
        best_fid = checkpoint.get('best_fid', float('inf'))
        print(f"Resumed from epoch {start_epoch}, iterations {iters}, best_fid {best_fid:.4f}")
    print("Starting Training Loop...")
    patience_counter = 0
    done = False
    for epoch in range(start_epoch, config['epochs']):
        if done: break
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{config['epochs']}",
                    initial=iters % len(dataloader))
        for i, data in pbar:
            if done: break
            time_data_start = time.time()
            netD.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            time_data_end = time.time()
            if config['r1_gamma'] > 0:
                real_cpu.requires_grad = True
            with autocast(enabled=config['amp']):
                real_aug = DiffAugment(real_cpu, policy=config['diffaug'])
                c = None
                if config['disc'] == 'pg':
                    output_real = netD(real_aug, c)
                    errD_real = (F.relu(torch.ones_like(output_real) - output_real)).mean()
                else:
                    output_real = netD(real_aug).view(-1)
                    errD_real = criterion(output_real, label)
            scaler.scale(errD_real).backward(create_graph=True)
            errD_r1 = torch.tensor(0.0, device=device)
            if config['r1_gamma'] > 0 and (i + 1) % config['r1_every'] == 0:
                with autocast(enabled=config['amp']):
                    grad_real = torch.autograd.grad(outputs=output_real.sum(), inputs=real_cpu, create_graph=True)[0]
                    r1_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                    errD_r1 = (config['r1_gamma'] / 2) * r1_penalty
                scaler.scale(errD_r1).backward()
            noise = torch.randn(b_size, config['zdim'], 1, 1, device=device)
            with autocast(enabled=config['amp']):
                fake = netG(noise)
                fake_aug = DiffAugment(fake.detach(), policy=config['diffaug'])
                if config['disc'] == 'pg':
                    output_fake = netD(fake_aug, c)
                    errD_fake = (F.relu(torch.ones_like(output_fake) + output_fake)).mean()
                else:
                    label.fill_(fake_label)
                    output_fake = netD(fake_aug).view(-1)
                    errD_fake = criterion(output_fake, label)
            scaler.scale(errD_fake).backward()
            errD = errD_real + errD_fake + errD_r1
            scaler.step(optimizerD)
            scaler.update()
            time_graph_start = time.time()
            if (i + 1) % config['n_critic'] == 0:
                netG.zero_grad()
                label.fill_(real_label)
                graph_stats = {}
                with autocast(enabled=config['amp']):
                    fake = netG(noise)
                    c = None
                    if config['disc'] == 'pg':
                        output = netD(DiffAugment(fake, policy=config['diffaug']), c)
                        errG_bce = (-output).mean()
                    else:
                        label.fill_(real_label)
                        output = netD(DiffAugment(fake, policy=config['diffaug'])).view(-1)
                        errG_bce = criterion(output, label)
                    errG = errG_bce
                    if config['graph'] == 'on':
                        graph_hook = NewGraphHook(segments=config['graph_segments'])
                        graph_loss, graph_stats = graph_hook.loss(fake)
                        errG += config['graph_w'] * graph_loss
                    loss_g_graph_diff, loss_g_graph_slic = torch.tensor(0.0), torch.tensor(0.0)
                    ms_diff, ms_slic, graph_l1_loss = 0.0, 0.0, 0.0
                time_graph_end = time.time()
                scaler.scale(errG).backward()
                scaler.step(optimizerG)
                scaler.update()
                with torch.no_grad():
                    ema_netG = netG_ema.module if hasattr(netG_ema, 'module') else netG_ema
                    main_netG = netG.module if hasattr(netG, 'module') else netG
                    for param_ema, param_main in zip(ema_netG.parameters(), main_netG.parameters()):
                        param_ema.data.mul_(config['ema_decay']).add_(param_main.data, alpha=1 - config['ema_decay'])
            else:
                time_graph_end = time_graph_start
            time_step_end = time.time()
            if i % 50 == 0:
                log_payload = {
                    'epoch': epoch + 1, 'iter': i,
                    'loss/d': errD.item(), 'loss/d_r1': errD_r1.item(),
                    'loss/g': errG.item() if 'errG' in locals() else 0,
                    'loss/g_bce': errG_bce.item() if 'errG_bce' in locals() else 0,
                    'loss/g_graph_diff': loss_g_graph_diff.item(),
                    'loss/g_graph_slic': loss_g_graph_slic.item(),
                    'graph/l1_loss': graph_l1_loss,
                    'lr/g': optimizerG.param_groups[0]['lr'],
                    'lr/d': optimizerD.param_groups[0]['lr'],
                    'time/data_ms': (time_data_end - time_data_start) * 1000,
                    'time/graph_ms_diff': ms_diff,
                    'time/graph_ms_slic': ms_slic,
                    'time/step_ms': (time_step_end - time_data_start) * 1000,
                    'vram/alloc_gb': torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0,
                    'vram/max_gb': torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0,
                    'pg/n_trainable_params': pg_n_trainable_params,
                    'pg/params_frozen': pg_params_frozen,
                }
                label_map_to_save = None
                if graph_stats:
                    scalar_stats = {k: v for k, v in graph_stats.items() if isinstance(v, (int, float))}
                    log_payload.update(scalar_stats)
                    if 'label_map' in graph_stats:
                        label_map_to_save = graph_stats['label_map']
                logger.log(log_payload, iters)
                pbar.set_postfix(
                    {k.split('/')[-1]: f'{v:.3f}' for k, v in log_payload.items() if isinstance(v, (int, float))})
                if label_map_to_save is not None:
                    map_norm = label_map_to_save.float() / label_map_to_save.max()
                    vutils.save_image(map_norm.unsqueeze(1), f"{config['out']}/graph_labelmap_step{iters:06d}.png",
                                      normalize=False)
            if (iters > 0 and
                    config['eval_every_iters'] > 0 and
                    iters % config['eval_every_iters'] == 0 and
                    iters >= config['eval_start_iters']):
                fid_val = run_periodic_eval(netG_ema, config, device, iters, val_path, fid_model)
                logger.log({'eval/fid': fid_val, 'epoch': epoch + 1}, iters)
                if fid_val < best_fid:
                    best_fid = fid_val
                    patience_counter = 0
                    g_state_dict = netG.module.state_dict() if hasattr(netG, 'module') else netG.state_dict()
                    g_ema_state_dict = netG_ema.module.state_dict() if hasattr(netG_ema,
                                                                               'module') else netG_ema.state_dict()
                    d_state_dict = netD.module.state_dict() if hasattr(netD, 'module') else netD.state_dict()
                    torch.save({
                        'generator_state_dict': g_state_dict, 'netG_ema_state_dict': g_ema_state_dict,
                        'discriminator_state_dict': d_state_dict, 'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(), 'scaler_state_dict': scaler.state_dict(),
                        'epoch': epoch, 'iters': iters, 'best_fid': best_fid, 'config': config,
                    }, f"{config['out']}/checkpoint_best.pt")
                    print(f"*** New best FID: {best_fid:.4f}. Saved checkpoint_best.pt ***")
                    best_score_data = {'best_fid': best_fid, 'step': iters}
                    with open(os.path.join(config['out'], 'best.json'), 'w') as f:
                        json.dump(best_score_data, f, indent=4)
                else:
                    patience_counter += 1
                    print(
                        f"FID did not improve. Best: {best_fid:.4f}. Patience: {patience_counter}/{config['early_stop_patience']}")
                if config['early_stop_patience'] > 0 and patience_counter >= config['early_stop_patience']:
                    print("Early stopping triggered.")
                    done = True
            if (iters % config['snap_iters'] == 0) or ((epoch == config['epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    netG_ema.eval()
                    fake_samples = netG_ema(fixed_noise).detach().cpu()
                vutils.save_image(fake_samples, f"{config['out']}/fakes_step{iters:06d}_ema.png", normalize=True)
                netG_ema.train()
            iters += 1
        with torch.no_grad():
            netG_ema.eval()
            fake_samples = netG_ema(fixed_noise).detach().cpu()
        vutils.save_image(fake_samples, f"{config['out']}/fakes_epoch_{epoch + 1:04d}_ema.png", normalize=True)
        netG_ema.train()
    logger.close()
    g_state_dict = netG.module.state_dict() if hasattr(netG, 'module') else netG.state_dict()
    g_ema_state_dict = netG_ema.module.state_dict() if hasattr(netG_ema, 'module') else netG_ema.state_dict()
    d_state_dict = netD.module.state_dict() if hasattr(netD, 'module') else netD.state_dict()
    torch.save({
        'generator_state_dict': g_state_dict,
        'netG_ema_state_dict': g_ema_state_dict,
        'discriminator_state_dict': d_state_dict,
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'iters': iters,
        'best_fid': best_fid,
        'config': config,
    }, f"{config['out']}/checkpoint_final.pt")
    print(f"Final checkpoint saved to {config['out']}/checkpoint_final.pt")
    print(f"Generating {config['n_gen']} fake images for evaluation...")
    fakes_dir = os.path.join(config['out'], 'fakes_ema')
    os.makedirs(fakes_dir, exist_ok=True)
    netG_ema.eval()
    with torch.no_grad():
        for i in tqdm(range(config['n_gen'])):
            noise = torch.randn(1, config['zdim'], 1, 1, device=device)
            with autocast(enabled=config['amp']):
                fake_img = netG_ema(noise)
            vutils.save_image(fake_img, os.path.join(fakes_dir, f'fake_{i:04d}.png'), normalize=True)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DCGAN Training Script")
    parser.add_argument('--dataroot', required=True, help='path to train dataset')
    parser.add_argument('--dataroot-val', type=str, default=None, help='path to validation dataset (for FID)')
    parser.add_argument('--out', type=str, default='output', help='folder to output images and model checkpoints')
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume training from')
    parser.add_argument('--seed', type=int, help='manual seed')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--zdim', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr-g', type=float, default=0.0001, help='learning rate for generator')
    parser.add_argument('--lr-d', type=float, default=0.0002, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam optimizer')
    parser.add_argument('--n-critic', type=int, default=1, help='number of discriminator updates per generator update')
    parser.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision')
    parser.add_argument('--ema-decay', type=float, default=0.999, help='decay for generator EMA')
    parser.add_argument('--diffaug', type=str, default='',
                        help='Policy for DiffAugment. E.g. "color,translation,cutout"')
    parser.add_argument('--r1-gamma', type=float, default=0.0, help='Gamma for R1 regularization. Set to 0 to disable.')
    parser.add_argument('--r1-every', type=int, default=16, help='Apply R1 regularization every N steps.')
    parser.add_argument('--no-tb', action='store_true', help='Disable TensorBoard logging')
    parser.add_argument('--snap-iters', type=int, default=1000, help='number of iterations between saving snapshots')
    parser.add_argument('--n-gen', type=int, default=512, help='number of final fakes to generate for final eval')
    parser.add_argument('--eval-every-iters', type=int, default=2000,
                        help='Run FID evaluation every N iterations. 0 to disable.')
    parser.add_argument('--eval-start-iters', type=int, default=0,
                        help='Start FID evaluation after N iterations.')
    parser.add_argument('--ref-name', type=str, default=None,
                        help='Name of the reference statistics for Clean-FID.')
    parser.add_argument('--eval-n-fakes', type=int, default=1024,
                        help='Number of fakes to generate for periodic FID evaluation')
    parser.add_argument('--early-stop-patience', type=int, default=5,
                        help='Number of evaluations without improvement to trigger early stopping. 0 to disable.')
    parser.add_argument('--disc', type=str, default='sg2', choices=['sg2', 'pg', 'sngan'],
                        help='Discriminator type: sg2 (DCGAN-style), pg (Projected-GAN), or sngan (SNGAN-ResNet).')
    parser.add_argument('--graph', type=str, default='off', choices=['on', 'off'],
                        help='Enable the graph hook for the generator.')
    parser.add_argument('--graph-w', type=float, default=0.0,
                        help='Weight for the graph hook loss (explicit opt-in).')
    parser.add_argument('--graph-segments', type=int, default=128,
                        help='Number of superpixels for the graph hook.')
    parser.add_argument('--legacy-graph-backend', type=str, choices=['diff', 'slic', 'both'], default='diff',
                        help='[Legacy] Graph regularizer backend.')
    parser.add_argument('--legacy-graph-w', type=float, default=1e-3,
                        help='[Legacy] Weight for the differentiable graph loss.')
    parser.add_argument('--legacy-graph-w-slic', type=float, default=0.0,
                        help='[Legacy] Weight for the SLIC-based monitoring loss.')
    parser.add_argument('--legacy-graph-subb', type=int, default=0,
                        help='[Legacy] Sub-batch size for graph loss (0 = off).')
    parser.add_argument('--legacy-graph-bins', type=int, default=32,
                        help='[Legacy] Number of bins for differentiable histogram.')
    parser.add_argument('--legacy-graph-cells', type=int, default=16,
                        help='[Legacy] Grid size for cell-based statistics.')
    parser.add_argument('--legacy-graph-down', type=int, default=2, help='[Legacy] Downsample factor for diff backend.')
    parser.add_argument('--legacy-slic-n', type=int, default=300, help='[Legacy] Number of SLIC segments.')
    parser.add_argument('--legacy-slic-compact', type=float, default=10.0, help='[Legacy] SLIC compactness.')
    parser.add_argument('--legacy-slic-sigma', type=float, default=1.0, help='[Legacy] SLIC gaussian sigma.')
    parser.add_argument('--legacy-slic-iters', type=int, default=10, help='[Legacy] SLIC iterations.')
    parser.add_argument('--legacy-slic-resize', type=int, default=0, help='[Legacy] Resize image for SLIC (0 = off).')
    args = parser.parse_args()
    config = vars(args)
    if config['graph'] == 'on':
        if config['graph_w'] <= 0:
            parser.error("--graph-w must be > 0 when --graph is on.")
        if config['disc'] != 'pg':
            print("Warning: The new graph hook is designed for and tested with --disc=pg.")
    if config['graph'] == 'on':
        if config['disc'] in ['pg', 'sngan'] and config['graph_w'] <= 0:
            parser.error(f"--graph-w must be > 0 when --graph is on for --disc={config['disc']}.")
        elif config['disc'] == 'sg2':
            print("Warning: --graph=on is specified, but this flag has no effect on the 'sg2' (DCGAN) backend.")
    main(config)
