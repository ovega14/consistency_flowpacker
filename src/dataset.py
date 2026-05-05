"""
Dataset for consistency distillation of FlowPacker.

Loads saved trajectory .pt files and returns pairs (x_t, x_{t+1}, t, protein_context).
"""
import math
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TrajectoryDataset(Dataset):
    """
    Wraps the saved FlowPacker trajectories into a PyTorch `Dataset` object.
    
    Each sample is a tuple (t, x_t, x_{t+1}, context) consisting of adjacent
    timestep pairs drawn from a saved trajectory.

    Note: During training, we should randomly sample a timestep index
    for each protein.

    Args:
        traj_dir (str): Path to the folder containing the `.pt` trajectory files
        n_steps (int): Number of timesteps in each trajectory. Default: `10`
        eps (float): Small temporal offset matching FlowPacker's `t_range`. Default: `2e-3`

    Attributes:
        self.files (list): Sorted list of all `.pt` files found in `traj_dir`
        self.t_range (Tensor): The `n_steps` t values matching FlowPacker's integration grid
    """
    def __init__(self, traj_dir: str, n_steps: int = 10, eps: float = 2e-3):
        self.traj_dir = Path(traj_dir)
        self.files = sorted(self.traj_dir.glob('*.pt'))
        assert len(self.files) > 0, f'No .pt files found in {traj_dir}'
        print(f'Loaded {len(self.files)} trajectory files from {traj_dir}')

        self.n_steps = n_steps
        self.t_range = torch.linspace(0, 1.0 - eps, n_steps)  # match FlowPacker's linspace

    def __len__(self) -> int:
        """Returns the total number of training samples."""
        # Each file contributes (n_steps - 1) training pairs
        return len(self.files) * (self.n_steps - 1)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a timestep pair given an index `idx`."""
        file_idx = idx // (self.n_steps - 1)
        step_idx = idx % (self.n_steps - 1)
        
        data = torch.load(self.files[file_idx], weights_only=False)

        chi_traj = data['chi_traj']
        aa = data['aa']
        atom_mask = data['atom_mask']
        chi_mask = data['chi_mask']
        bb_coords = data['bb_coords']

        x_t = chi_traj[step_idx]  # [n_res, 4]
        x_t1 = chi_traj[step_idx + 1]  # [n_res, 4]
        t = self.t_range[step_idx]  # scalar

        return {
            'x_t' : x_t,
            'x_t1' : x_t1,
            't' : t,
            'aa' : aa,
            'atom_mask' : atom_mask,
            'chi_mask' : chi_mask,
            'bb_coords' : bb_coords
        }  # TODO(OV): Modify this to allow linear interpolation for multi-step sampling


def _test_trajectory_dataset():
    print('[Testing TrajectoryDataset...]')
    traj_dir = '/u/octavio5/projects/consistency_flowpacker/flowpacker/samples/traj-all/run_1'
    dataset = TrajectoryDataset(traj_dir, n_steps=10)

    # Test length
    expected_len = len(dataset.files) * 9
    assert len(dataset) == expected_len, '[FAILED: len(dataset) incorrect]'

    # Test single sample shapes
    sample = dataset[0]
    n_res = sample['x_t'].shape[0]
    assert sample['x_t'].shape == (n_res, 4), f'[FAIL] x_t shape: {sample["x_t"].shape}'
    assert sample['x_t1'].shape == (n_res, 4), f'[FAIL] x_t1 shape: {sample["x_t1"].shape}'
    assert sample['t'].shape == (), f'[FAIL] t should be scalar, got {sample["t"].shape}'
    assert sample['aa'].shape == (n_res,), f'[FAIL] aa shape: {sample["aa"].shape}'
    assert sample['bb_coords'].shape == (n_res, 4, 3), f'[FAIL] bb_coords shape: {sample["bb_coords"].shape}'
    assert sample['chi_mask'].shape == (n_res, 4), f'[FAIL] chi_mask shape: {sample["chi_mask"].shape}'
    assert sample['atom_mask'].shape == (n_res, 14), f'[FAIL] atom_mask shape: {sample["atom_mask"].shape}'
    print(f'[PASS] all shapes correct for sample with {n_res} residues')

    # Test x_t and x_t1 are different (adjacent timesteps)
    assert not torch.allclose(sample['x_t'], sample['x_t1']), '[FAIL] x_t and x_t1 are identical'
    print('[PASS] x_t and x_t1 are different')

    # Test t values match expected range
    sample_last = dataset[8]  # last pair of first file
    assert dataset.t_range[0] == sample['t'], '[FAIL] t value mismatch at step 0'
    assert dataset.t_range[8] == sample_last['t'], '[FAIL] t value mismatch at step 8'
    print(f'[PASS] t values correct: first={sample["t"]:.3f}, last={sample_last["t"]:.3f}')

    # Test chi angles in [0, 2pi] where mask is 1
    chi_mask = sample['chi_mask']
    active = sample['x_t'][chi_mask == 1]
    assert (active >= 0).all() and (active <= 2*math.pi).all(), '[FAIL] x_t out of [0, 2pi]'
    print('[PASS] chi angles in [0, 2pi]')

    print('[DONE]')


if __name__ == '__main__': _test_trajectory_dataset()
