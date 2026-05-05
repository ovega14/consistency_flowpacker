"""Various helper functions."""
import math
import torch
import sys

# flowpacker imports
sys.path.insert(0, '/u/octavio5/projects/consistency_flowpacker/flowpacker')
from utils.sidechain_utils import get_bb_dihedral


def grab(inp):
    """Detaches tensor from computational graph and returns as NumPy array."""
    if hasattr(inp, 'detach'):
        return inp.detach().cpu().numpy()
    return inp


def wrap(inp):
    """Wraps an input into the interval [0, 2\pi)."""
    return (inp + math.pi) % (2*math.pi) - math.pi


@torch.no_grad()
def update_ema(ema_model, online_model, mu=0.95) -> None:
    """
    Updates the parameters of the ema_model according to
    EMA update: theta_ema = mu * theta_ema + (1-mu) * theta_online

    Args:
        ema_model (Module): Consistency model with EMA parameters
        online_model (Module): Consistency model with updating parameters
        mu (float): EMA decay factor. Default: 0.95
    """
    for ema_p, online_p in zip(ema_model.parameters(), online_model.parameters()):
        ema_p.data.mul_(mu).add_(online_p.data, alpha=1 - mu)


def get_aa_onehot_and_dihedral(batch_dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Converts amino acid indices to one-hot and computes backbone dihedrals
    from backbone coordinates. 
    
    Note: Number of amino acid classes is 21 (= 20 known + 1 unknown).

    Args:
        batch_dict (dict): Dictionary containing all data for a batch of proteins
        device (torch.device): PyTorch device being used

    Returns:
        aa_onehot (Tensor): One-hot encoding of amino acids [total_res, 21]
        bb_dihedral (Tensor): Three backbone dihedral angles [total_res, 3]
    """
    aa = batch_dict['aa'].to(device)  # [total_res]
    bb_coords = batch_dict['bb_coords'].to(device)  # [total_res, 4, 3]
    batch_id = batch_dict['batch_id'].to(device)
 
    # One-hot encode amino acids (20 standard AAs)
    aa_onehot = torch.zeros(aa.shape[0], 21, device=device)
    aa_onehot.scatter_(1, aa.unsqueeze(1).clamp(0, 20), 1.0)
 
    # Backbone dihedrals (phi, psi, omega) from N, CA, C, O coords
    bb_dihedral = get_bb_dihedral(bb_coords[:, 0], bb_coords[:, 1], bb_coords[:, 2])
 
    return aa_onehot, bb_dihedral  # NOTE(OV): These are needed as node features


def _test_get_aa_onehot_and_dihedral():
    print('[Testing get_aa_onehot_and_dihedral...]')
    device = torch.device('cpu')
    n_res = 50

    batch_dict = {
        'aa' : torch.randint(0, 21, (n_res,)),
        'bb_coords' : torch.randn(n_res, 4, 3),
        'batch_id' : torch.cat([torch.zeros(25, dtype=torch.long), torch.ones(25, dtype=torch.long)]),
    }

    aa_onehot, bb_dihedral = get_aa_onehot_and_dihedral(batch_dict, device)

    # shapes
    assert aa_onehot.shape == (n_res, 21), f'[FAIL] aa_onehot shape: {aa_onehot.shape}'
    assert bb_dihedral.shape == (n_res, 3), f'[FAIL] bb_dihedral shape: {bb_dihedral.shape}'
    print(f'[PASS] aa_onehot shape: {aa_onehot.shape}')
    print(f'[PASS] bb_dihedral shape: {bb_dihedral.shape}')

    # aa_onehot is valid one-hot: each row sums to 1, values are 0 or 1
    assert (aa_onehot.sum(dim=1) == 1).all(), '[FAIL] aa_onehot rows do not sum to 1'
    assert ((aa_onehot == 0) | (aa_onehot == 1)).all(), '[FAIL] aa_onehot contains non-binary values'
    print('[PASS] aa_onehot is valid one-hot')

    # bb_dihedral values are in [-pi, pi]
    assert (bb_dihedral >= -math.pi).all() and (bb_dihedral <= math.pi).all(), '[FAIL] bb_dihedral out of range'
    print('[PASS] bb_dihedral values in [-pi, pi]')

    # device
    assert aa_onehot.device == device, '[FAIL] aa_onehot on wrong device'
    assert bb_dihedral.device == device, '[FAIL] bb_dihedral on wrong device'
    print('[PASS] tensors on correct device')


if __name__ == '__main__': _test_get_aa_onehot_and_dihedral()
