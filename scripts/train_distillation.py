"""Script to train Riemannian consistency model via distillation of flowpacker model."""
import math
import torch
import numpy as np

import yaml
import copy
import tqdm.auto as tqdm
from easydict import EasyDict as edict
from pathlib import Path

import sys
sys.path.insert(0, '../src/')
import consistency
from dataset import get_consistency_dataloader
from util import wrap, get_aa_onehot_and_dihedral, update_ema


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


def train(
    traj_dir: str = '../flowpacker/samples/traj-all/run_1',
    config_path: str = '/u/octavio5/projects/consistency_flowpacker/flowpacker/config/training/vf.yaml',
    ckpt_path: str= '../flowpacker/checkpoints/bc40.pth',
    epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-4,
    ema_mu: float = 0.95,
    save_interval: int = 10,
    save_dir: str= '../checkpoints/consistency',
    model_type: str= 'MPConsistencyModel'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = edict(yaml.load(open(config_path), Loader=yaml.FullLoader))

    # Build models
    if model_type == 'EquiformerConsistencyModel':
        online_model = build_consistency_model(config, device)  # load FlowPacker checkpoint
    else:
        online_model = getattr(consistency, model_type)().to(device)
    ema_model = copy.deepcopy(online_model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Initialize from FlowPacker checkpoint
    if ckpt_path and model_type == 'EquiformerConsistencyModel':
        print(f'Loading FlowPacker checkpoint from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        #state_dict = ckpt.get('model', ckpt.get('ema_model', ckpt))
        state_dict = {k.replace('module.model.', ''): v for k, v in ckpt['state_dict'].items()}
        missing, unexpected = online_model.model.load_state_dict(state_dict, strict=False)
        print(f'  Missing: {len(missing)}, Unexpected: {len(unexpected)}')
        ema_model = copy.deepcopy(online_model)

    # Dataloader
    loader = get_consistency_dataloader(traj_dir, device=device, batch_size=batch_size, shuffle=True, num_workers=0)

    #t_range = torch.linspace(0, 1.0 - config.sample.eps, config.sample.num_steps).to(device)
    t_range = torch.linspace(0, 1.0 - 2e-3, 10).to(device)

    optimizer = torch.optim.AdamW(online_model.parameters(), lr=lr, weight_decay=1e-12)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    losses = []
    for epoch in tqdm.tqdm(range(epochs)):
        online_model.train()
        total_loss = 0.
        for batch_dict in tqdm.tqdm(loader, desc=f'Epoch {epoch+1}/{epochs}', leave=False):
            loss = train_step(batch_dict, t_range, online_model, ema_model, ema_mu, optimizer, device)
            total_loss += loss
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, lr={scheduler.get_last_lr()[0]:.2e}')

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            ckpt_out = Path(save_dir) / f"consistency_ep{epoch+1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model': online_model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_out)
            print(f"  Saved {ckpt_out}")

    return online_model, ema_model, losses


def main():
    online_model, ema_model, losses = train(
        traj_dir='../flowpacker/samples/traj-500/run_1',
        epochs=100,
        batch_size=64,
        lr=1e-3,
        ema_mu=0.99,
        save_interval=20,
    )


if __name__ == '__main__':
    main()
