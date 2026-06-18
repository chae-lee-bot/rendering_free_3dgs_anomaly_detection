import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from configs import Config
from dataset import NormalGaussianDataset, ATTR_SLICES, compute_normalizer
from mae_model import GaussianMAE


def compute_loss(recon, target, weights, loss_type='l2', eps=1e-3):
    """
    loss_type:
      'l2'         : 기존 L2 loss
      'charbonnier': Charbonnier loss (L1과 L2의 중간, 미세 차이에 더 민감)
                     loss = sqrt((pred - target)^2 + eps^2) -> 근데 이게 성능이 더 별로였음. 일단 남겨둠! 
    """
    loss_dict = {}
    total = 0.0
    
    for name in ['xyz', 'f_dc', 'opacity', 'scale']:
        sl = ATTR_SLICES[name]
        diff = recon[..., sl] - target[..., sl]
        
        if loss_type == 'charbonnier':
            # Charbonnier: 작은 error에도 gradient 유지
            loss = torch.sqrt(diff ** 2 + eps ** 2).mean()
        else:
            # L2 (기존)
            loss = (diff ** 2).mean()
        
        weighted = loss * weights[name]
        loss_dict[name] = loss.item()
        total = total + weighted
    
    # Quaternion loss
    sl = ATTR_SLICES['rotation']
    q_pred = recon[..., sl]
    q_gt = target[..., sl]
    q_pred_n = q_pred / (q_pred.norm(dim=-1, keepdim=True) + 1e-6)
    q_gt_n = q_gt / (q_gt.norm(dim=-1, keepdim=True) + 1e-6)
    dot = (q_pred_n * q_gt_n).sum(dim=-1).abs()
    loss_q = (1 - dot).mean()
    weighted_q = loss_q * weights['rotation']
    loss_dict['rotation'] = loss_q.item()
    total = total + weighted_q
    
    loss_dict['total'] = total.item()
    return total, loss_dict


def get_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def train(cfg):
    os.makedirs(cfg.train.output_dir, exist_ok=True)
    
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    
    device = cfg.train.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Normalizer 계산 + 저장
    normalizer = compute_normalizer(
        data_root=cfg.data.data_root,
        classes=cfg.data.classes,
        n_samples_per_ply=5000,
        seed=cfg.train.seed,
    )
    norm_path = os.path.join(cfg.train.output_dir, 'normalizer.npz')
    normalizer.save(norm_path)
    
    # 데이터셋 (정규화 적용)
    dataset = NormalGaussianDataset(
        data_root=cfg.data.data_root,
        classes=cfg.data.classes,
        n_gaussians=cfg.data.n_gaussians,
        normalizer=normalizer,
        seed=cfg.train.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # 모델
    model = GaussianMAE(
        num_groups=cfg.model.num_groups,
        group_size=cfg.model.group_size,
        input_dim=cfg.data.input_dim,
        embed_dim=cfg.model.embed_dim,
        encoder_depth=cfg.model.encoder_depth,
        decoder_depth=cfg.model.decoder_depth,
        num_heads=cfg.model.num_heads,
        mlp_ratio=cfg.model.mlp_ratio,
        mask_ratio=cfg.model.mask_ratio,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model 파라미터 수: {n_params:,}")
    
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        betas=(0.9, 0.95),
    )
    
    loss_weights = {
        'xyz':      cfg.model.loss_weight_xyz,
        'f_dc':     cfg.model.loss_weight_f_dc,
        'opacity':  cfg.model.loss_weight_opacity,
        'scale':    cfg.model.loss_weight_scale,
        'rotation': cfg.model.loss_weight_rotation,
    }
    print(f"Loss weights: {loss_weights}")
    
    steps_per_epoch = len(loader)
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = cfg.train.warmup_epochs * steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")
    
    print(f"\n{'=' * 60}")
    print(f"학습 시작")
    print(f"{'=' * 60}\n")
    
    global_step = 0
    start_time = time.time()
    
    for epoch in range(cfg.train.epochs):
        model.train()
        epoch_losses = {'total': 0, 'xyz': 0, 'f_dc': 0, 'opacity': 0, 'scale': 0, 'rotation': 0}
        n_batches = 0
        
        for batch in loader:
            params = batch['params'].to(device, non_blocking=True)
            
            lr = get_lr(global_step, total_steps, warmup_steps, cfg.train.lr)
            for g in optimizer.param_groups:
                g['lr'] = lr
            
            recon, target, mask, centers, patch_idx = model(params)
            
            loss, loss_dict = compute_loss(recon, target, loss_weights, 
                                            loss_type=getattr(cfg.train, 'loss_type', 'l2'))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1
            global_step += 1
        
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        
        elapsed = time.time() - start_time
        print(f"[Epoch {epoch+1:3d}/{cfg.train.epochs}] "
              f"loss={epoch_losses['total']:.4f} "
              f"(xyz={epoch_losses['xyz']:.4f}, "
              f"f_dc={epoch_losses['f_dc']:.4f}, "
              f"opa={epoch_losses['opacity']:.4f}, "
              f"sca={epoch_losses['scale']:.4f}, "
              f"rot={epoch_losses['rotation']:.4f}) "
              f"lr={lr:.2e} "
              f"time={elapsed:.0f}s")
        
        if (epoch + 1) % cfg.train.save_interval == 0 or (epoch + 1) == cfg.train.epochs:
            ckpt_path = os.path.join(cfg.train.output_dir, f'mae_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'loss': epoch_losses['total'],
            }, ckpt_path)
            print(f"  ✅ 저장: {ckpt_path}")
    
    final_path = os.path.join(cfg.train.output_dir, 'mae_final.pt')
    torch.save({
        'epoch': cfg.train.epochs,
        'model_state_dict': model.state_dict(),
        'config': cfg,
    }, final_path)
    print(f"\n 학습 완료! 최종 모델: {final_path}")
    print(f"   Normalizer: {norm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--n_gaussians', type=int, default=None)
    parser.add_argument('--classes', nargs='+', default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--loss_type', type=str, default='l2',
                        choices=['l2', 'charbonnier'],
                        help='Loss 함수 종류 (기본: l2)')
    parser.add_argument('--embed_dim', type=int, default=None,
                        help='Transformer embedding dim (기본 configs: 384)')
    parser.add_argument('--encoder_depth', type=int, default=None,
                        help='Encoder layer 수 (기본 configs: 6)')
    parser.add_argument('--decoder_depth', type=int, default=None,
                        help='Decoder layer 수 (기본 configs: 2)')
    parser.add_argument('--num_heads', type=int, default=None,
                        help='Attention head 수 (기본 configs: 6)')
    args = parser.parse_args()
    
    cfg = Config()
    if args.epochs: cfg.train.epochs = args.epochs
    if args.batch_size: cfg.train.batch_size = args.batch_size
    if args.lr: cfg.train.lr = args.lr
    if args.n_gaussians: cfg.data.n_gaussians = args.n_gaussians
    if args.classes: cfg.data.classes = args.classes
    if args.output_dir: cfg.train.output_dir = args.output_dir
    if args.data_root: cfg.data.data_root = args.data_root
    cfg.train.loss_type = args.loss_type
    
    # 모델 크기 변경
    if args.embed_dim: cfg.model.embed_dim = args.embed_dim
    if args.encoder_depth: cfg.model.encoder_depth = args.encoder_depth
    if args.decoder_depth: cfg.model.decoder_depth = args.decoder_depth
    if args.num_heads: cfg.model.num_heads = args.num_heads
    
    print(f"Loss type: {cfg.train.loss_type}")
    print(f"Model: embed={cfg.model.embed_dim}, "
          f"enc={cfg.model.encoder_depth}, dec={cfg.model.decoder_depth}, "
          f"heads={cfg.model.num_heads}")
    
    print(cfg)
    train(cfg)