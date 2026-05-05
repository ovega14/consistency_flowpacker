"""Script to evaulate trained Riemannian consistency model."""
import math
import random
import torch
import numpy as np
import argparse
from pathlib import Path

import sys
sys.path.insert(0, '../src/')
from fabrics import make_consistency_models
from util import get_aa_onehot_and_dihedral, wrap


def evaluate(
    traj_dir: str = '../flowpacker/samples/traj-500/run_1',
    ckpt_path: str = '../checkpoints/consistency/consistency_ep100.pt',
    model_type: str = 'ConditionedMPConsistencyModel',
    config_path: str = '/u/octavio5/projects/consistency_flowpacker/flowpacker/config/training/vf.yaml',
    n_test: int = 100,
    seed: int = 42,
) -> None:
    """Docs TODO"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load model
    online_model, _ = make_consistency_models(model_type, device, config_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    online_model.load_state_dict(ckpt['model'])
    online_model.eval()
    print(f'Loaded {model_type} from {ckpt_path} (epoch {ckpt["epoch"]})')

    # Split files
    random.seed(seed)
    files = sorted(Path(traj_dir).glob('*.pt'))
    random.shuffle(files)
    test_files = files[:n_test]
    print(f'Using {len(test_files)} proteins as test set ({len(files) - len(test_files)} train, {len(test_files)} test)')

    # Count residues
    total_residues = sum(
        torch.load(f, weights_only=False)['chi_mask'].sum().item()
        for f in test_files
    )
    print(f'Test set: {int(total_residues)} valid residues across {n_test} proteins')

    all_mae, all_acc = [], []
    per_chi_mae = [[] for _ in range(4)]
    per_chi_acc = [[] for _ in range(4)]
    per_chi_n = [0] * 4
    processed = 0

    for f in test_files:
        data = torch.load(f, weights_only=False)

        chi_traj = data['chi_traj'].to(device)
        chi_mask = data['chi_mask'].to(device)
        bb_coords = data['bb_coords'].to(device)
        aa = data['aa'].to(device)
        atom_mask = data['atom_mask'].to(device)

        x0 = chi_traj[0]
        x1_teacher = chi_traj[-1]
        batch_id = torch.zeros(aa.shape[0], dtype=torch.long, device=device)

        aa_onehot, bb_dihedral = get_aa_onehot_and_dihedral(
            {'aa': aa, 'bb_coords': bb_coords, 'batch_id': batch_id}, device
        )
        batch_dict = {
            'chi_mask': chi_mask, 'aa_onehot': aa_onehot,
            'bb_dihedral': bb_dihedral, 'atom_mask': atom_mask,
            'bb_coords': bb_coords, 'batch_id': batch_id, 'aa': aa,
        }

        t0 = torch.zeros(aa.shape[0], 1, device=device)
        with torch.no_grad():
            x1_pred = online_model(x0, t0, batch_dict)

        # Per chi metrics
        for chi_idx in range(4):
            mask = chi_mask[:, chi_idx].bool()
            if mask.sum() == 0:
                continue
            pred = x1_pred[mask, chi_idx]
            target = x1_teacher[mask, chi_idx]
            diff = wrap(pred - target)
            mae_deg = diff.abs().mean().item() * 180 / math.pi
            acc = (diff.abs() < (20 * math.pi / 180)).float().mean().item() * 100
            per_chi_mae[chi_idx].append(mae_deg)
            per_chi_acc[chi_idx].append(acc)
            per_chi_n[chi_idx] += mask.sum().item()

        # Overall
        diff_all = wrap(x1_pred - x1_teacher) * chi_mask
        mae_all = diff_all.abs().sum() / chi_mask.sum()
        acc_all = ((diff_all.abs() < (20 * math.pi / 180)) * chi_mask).sum() / chi_mask.sum()
        all_mae.append(mae_all.item() * 180 / math.pi)
        all_acc.append(acc_all.item() * 100)

        processed += chi_mask.sum().item()
        if int(processed) % 5000 < chi_mask.sum().item():
            print(f'  Processed {int(processed)}/{int(total_residues)} residues...')

    print()
    print('=' * 55)
    print(f'Evaluation Results (1-step sampling)')
    print('=' * 55)
    print(f'\nOverall (all chi angles):')
    print(f'  Angle MAE      : {np.mean(all_mae):.2f}°')
    print(f'  Angle Accuracy : {np.mean(all_acc):.2f}%  (within 20°)')
    print(f'\nPer-chi breakdown:')
    print(f'  {"Chi":<8} {"MAE (°)":<12} {"Accuracy (%)":<16} {"N valid"}')
    print(f'  {"-"*50}')
    for i in range(4):
        if per_chi_n[i] > 0:
            print(f'  chi{i+1:<5} {np.mean(per_chi_mae[i]):<12.2f} {np.mean(per_chi_acc[i]):<16.2f} {per_chi_n[i]}')
    print('=' * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', type=str,
                        default='../flowpacker/samples/traj-500/run_1')
    parser.add_argument('--ckpt_path', type=str,
                        default='../checkpoints/consistency/consistency_ep100.pt')
    parser.add_argument('--model_type', type=str,
                        default='ConditionedMPConsistencyModel')
    parser.add_argument('--config_path', type=str,
                        default='/u/octavio5/projects/consistency_flowpacker/flowpacker/config/training/vf.yaml')
    parser.add_argument('--n_test', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    evaluate(**vars(args))
