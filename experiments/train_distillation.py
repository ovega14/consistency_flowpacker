"""Script to train Riemannian consistency model via distillation of flowpacker model."""
import math
import torch
import numpy as np

import yaml
import copy
import argparse
import tqdm.auto as tqdm
from easydict import EasyDict as edict
from pathlib import Path

import sys
sys.path.insert(0, '../src/')
import consistency
from dataset import get_consistency_dataloader
from util import wrap, get_aa_onehot_and_dihedral, update_ema
from fabrics import make_consistency_models, make_scheduler


def train_step(batch_dict, t_range, online_model, ema_model, ema_mu, optimizer, device):
    """Executes a single step of consistency distillation training."""
    for k in batch_dict:
        batch_dict[k] = batch_dict[k].to(device)

    aa_onehot, bb_dihedral = get_aa_onehot_and_dihedral(batch_dict, device)
    batch_dict['aa_onehot'] = aa_onehot
    batch_dict['bb_dihedral'] = bb_dihedral

    t_vals = batch_dict['t'][:, 0]
    t_idx = torch.argmin(torch.abs(t_vals.unsqueeze(1) - t_range.unsqueeze(0)), dim=1)
    t_idx = t_idx.clamp(0, len(t_range) - 2)

    x_t = batch_dict['x_t']
    x_t1 = batch_dict['x_t1']
    chi_mask = batch_dict['chi_mask']
    t = t_range[t_idx].unsqueeze(-1)
    t1 = t_range[(t_idx + 1).clamp(max=len(t_range) - 1)].unsqueeze(-1)

    optimizer.zero_grad()
    x1_pred = online_model(x_t, t, batch_dict)
    with torch.no_grad():
        x1_target = ema_model(x_t1, t1, batch_dict)

    diff = wrap(x1_pred - x1_target)
    loss = torch.sum(diff**2) / torch.sum(chi_mask).clamp(min=1)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_model.parameters(), 1.0)
    optimizer.step()
    update_ema(ema_model, online_model, mu=ema_mu)

    return loss.item()


def train(args):
    """Runs the full consistency distillation training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    online_model, ema_model = make_consistency_models(args.model_type, device, args.config_path)

    loader = get_consistency_dataloader(
        args.traj_dir, device=device, batch_size=args.batch_size,
        shuffle=True, num_workers=0
    )

    t_range = torch.linspace(0, 1.0 - 2e-3, 10).to(device)

    optimizer = torch.optim.AdamW(online_model.parameters(), lr=args.lr, weight_decay=1e-12)
    scheduler = make_scheduler(args.scheduler, optimizer, T_max=args.epochs)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    losses = []
    for epoch in tqdm.tqdm(range(args.epochs)):
        online_model.train()
        total_loss = 0.
        for batch_dict in tqdm.tqdm(loader, desc=f'Epoch {epoch+1}/{args.epochs}', leave=False):
            loss = train_step(batch_dict, t_range, online_model, ema_model, args.ema_mu, optimizer, device)
            total_loss += loss
        scheduler.step()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}')

        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            ckpt_out = Path(args.save_dir) / f'consistency_ep{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model': online_model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_out)
            print(f'  Saved {ckpt_out}')

    return online_model, ema_model, losses


def main():
    parser = argparse.ArgumentParser(description='Train Riemannian consistency model via distillation')
    parser.add_argument('--traj_dir', type=str,
                        default='../flowpacker/samples/traj-500/run_1')
    parser.add_argument('--config_path', type=str,
                        default='/u/octavio5/projects/consistency_flowpacker/flowpacker/config/training/vf.yaml')
    parser.add_argument('--ckpt_path', type=str,
                        default='../flowpacker/checkpoints/bc40.pth')
    parser.add_argument('--model_type', type=str,
                        default='ConditionedMPConsistencyModel',
                        choices=['EquiformerConsistencyModel', 'MPConsistencyModel', 'ConditionedMPConsistencyModel', 'ConditionedMPConsistencyModelV2'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--ema_mu', type=float, default=0.99)
    parser.add_argument('--scheduler', type=str, default='ShadowScheduler')
    parser.add_argument('--save_interval', type=int, default=20)
    parser.add_argument('--save_dir', type=str,
                        default='../checkpoints/consistency')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
