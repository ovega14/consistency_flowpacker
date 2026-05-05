"""Some pre-baked consistency models."""
import math
import torch
from torch_cluster import knn_graph, radius_graph

import sys
import yaml
from easydict import EasyDict as edict

# flowpacker imports
sys.path.insert(0, '/u/octavio5/projects/consistency_flowpacker/flowpacker')
from models.equiformer_v2.equiformer_v2 import EquiformerV2
from dataset_cluster import get_edge_features
from utils.sidechain_utils import Idealizer, get_bb_dihedral


if __name__ == '__main__':  # set device for testing
    HAS_CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if HAS_CUDA else 'cpu')


class EquiformerConsistencyModel(torch.nn.Module):
    """
    Wraps EquiformerV2 to act as a consistency model f(x_t, t) -> x_1.

    Note: The boundary condition f(x_1, t=1) = x_1 ensures that f is the 
    identity at t = 1. This is enforced by the skip connection `c_skip(t)`.

    Following the consistency model paper 2303.01469, parameterization is:
        
        f(x, t) = c_skip(t) * x + c_out(t) * F_theta(x, t)

    where c_skip(t) -> 1 and c_out(t) -> 0 as t -> 1.

    Args:
        model: Instance of EquiformerV2
        config: Config params
        eps (float): Small temporal offset. Default: `2e-3`
        sigma_min (float): Scale for skip connection and output weights
    """
    def __init__(self, model, config, eps=2e-3, sigma_min=0.002):
        super().__init__()
        self.model = model  # EquiformerV2
        self.config = config
        self.eps = eps
        self.sigma_min = sigma_min
        self.idealizer = Idealizer(use_native_bb_coords=True)

    def c_skip(self, t):
        """Skip connection weight. -> 1 as t -> 1."""
        return self.sigma_min**2 / ((t - 1.0)**2 + self.sigma_min**2)

    def c_out(self, t):
        """Output weight. -> 0 as t -> 1."""
        return (self.sigma_min * (1 - t)) / (self.sigma_min**2 + (t - 1)**2)**0.5

    def get_node_edge_features(self, xt, bb_coords, aa_onehot, bb_dihedral, atom_mask, batch_id):
        """Builds node/edge features similar to FlowPacker's get_vf."""
        # Node features: aa one-hot + backbone dihedrals
        dihedral_sin_cos = torch.cat([torch.sin(bb_dihedral), torch.cos(bb_dihedral)], dim=-1)
        node_feat = torch.cat([aa_onehot, dihedral_sin_cos], dim=-1)
 
        # Noised coordinates via idealizer
        x_scaled = (xt - math.pi) * (xt != 0).float()  # chi_mask applied upstream
        noised_crds = self.idealizer(
            torch.argmax(aa_onehot, dim=-1) if aa_onehot.shape[-1] > 1 else aa_onehot.squeeze(-1),
            bb_coords[:, :4],
            x_scaled
        ) * atom_mask.unsqueeze(-1)
 
        # Virtual Cb node coordinates (matching FlowPacker)
        if self.config.model.use_virtual_cb:
            b = bb_coords[:, 1] - bb_coords[:, 0]
            c = bb_coords[:, 2] - bb_coords[:, 1]
            a = torch.cross(b, c, dim=-1)
            node_crds = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + bb_coords[:, 1]
        else:
            node_crds = bb_coords[:, 1]  # CA
 
        # Edges
        if self.config.data.edge_type == 'knn':
            edge_index = knn_graph(node_crds, k=self.config.data.max_neighbors, batch=batch_id)
        else:
            edge_index = radius_graph(node_crds, r=self.config.data.max_radius,
                                      max_num_neighbors=self.config.data.max_neighbors,
                                      batch=batch_id)
 
        edge_feat = get_edge_features(noised_crds, edge_index, atom_mask,
                                      all_atoms=self.config.model.add_dist_to_edge)
 
        return node_feat, node_crds, edge_index, edge_feat, noised_crds

    def forward(self, xt, t, batch_dict) -> torch.Tensor:
        """
        Forward pass of consistency model.
        
        Args:
            xt (Tensor): `[total_res, 4]` chi angles at time t
            t (Tensor): `[total_res, 1]` timestep per residue
            batch_dict: dict with `aa_onehot`, `bb_dihedral`, `atom_mask`, etc.
 
        Returns:
            x1_pred (Tensor): `[total_res, 4]` predicted chi angles at t=1
        """
        aa_onehot = batch_dict['aa_onehot']
        bb_dihedral = batch_dict['bb_dihedral']
        atom_mask = batch_dict['atom_mask']
        chi_mask = batch_dict['chi_mask']
        batch_id = batch_dict['batch_id']
        bb_coords = batch_dict['bb_coords']
 
        node_feat, node_crds, edge_index, edge_feat, _ = self.get_node_edge_features(
            xt, bb_coords, aa_onehot, bb_dihedral, atom_mask, batch_id
        )
 
        # Raw network output
        F_out = self.model(t, xt, node_feat, node_crds, edge_index, edge_feat,
                           chi_mask, batch_id, atom_mask=atom_mask, 
                           self_cond=None)
 
        # Consistency parameterization: f = c_skip * x + c_out * F
        # t is per-residue [total_res, 1], broadcast over chi dim
        t_scalar = t[:, 0:1]  # [total_res, 1]
        skip = self.c_skip(t_scalar)
        out = self.c_out(t_scalar)
 
        # Riemannian: use exp map on torus for the skip + output combination
        # in tangent space, then project
        x1_pred = skip * xt + out * F_out
        x1_pred = x1_pred * chi_mask
        x1_pred = torch.remainder(x1_pred, 2 * math.pi)
        return x1_pred


def _test_equiformer_consistency_model():
    print('[Testing EquiformerConsistencyModel...]')
    n_res = 50  # total number of residues across all proteins in the batch
    n_batch = 2  # number of protiens in the batch
    batch_id = torch.cat([torch.zeros(25, dtype=torch.long), torch.ones(25, dtype=torch.long)])  # integer tensor saying which protein each residue belongs to (0 or 1)
    aa = torch.randint(0, 20, (n_res,))  # integer amino acid type for each residue
    aa_onehot = torch.zeros(n_res, 21)  
    aa_onehot.scatter_(1, aa.unsqueeze(1), 1.0)  # one-hot encoding of aa
    chi_mask = torch.randint(0, 2, (n_res, 4)).float()  # binary mask saying which of the 4 chi angles exist for each residue
    atom_mask = torch.ones(n_res, 14)  # binary mask saying which of the 14 possible heavty atoms exist for each residue
    bb_coords = torch.randn(n_res, 4, 3)  # [n_res, 4, 3] 3D coordinates of 4 backbone atoms (N, CA, C, O)
    bb_dihedral = torch.randn(n_res, 3)  # 3 backbone dihedral angles
    x_t = 2*math.pi * torch.rand(n_res, 4) * chi_mask  # the 4 chi angles for each residue at timestep t
    print(f'{x_t.shape=}')
    print(f'{x_t.device=}')

    batch_dict = {
        'aa_onehot': aa_onehot, 'bb_dihedral': bb_dihedral,
        'atom_mask': atom_mask, 'chi_mask': chi_mask,
        'bb_coords': bb_coords, 'batch_id': batch_id,
        'aa': aa
    }
    config = edict(
        yaml.load(open('/u/octavio5/projects/consistency_flowpacker/flowpacker/config/training/vf.yaml'),
        Loader=yaml.FullLoader)
    )
    config.seed = 1234

    equiformer = EquiformerV2(
        node_feature_in=config.model.node_feature_in,
        edge_feature_in=config.model.edge_feature_in,
        num_layers=config.model.num_layers,
        lmax_list=config.model.lmax_list,
        mmax_list=config.model.mmax_list,
        sphere_channels=config.model.sphere_channels,
        attn_hidden_channels=config.model.attn_hidden_channels,
        num_heads=config.model.num_heads,
        attn_alpha_channels=config.model.attn_alpha_channels,
        attn_value_channels=config.model.attn_value_channels,
        ffn_hidden_channels=config.model.ffn_hidden_channels,
        edge_channels=config.model.edge_channels,
        share_atom_edge_embedding=config.model.share_atom_edge_embedding,
        use_atom_edge_embedding=config.model.use_atom_edge_embedding,
        attn_activation=config.model.attn_activation,
        ffn_activation=config.model.ffn_activation,
        use_gate_act=config.model.use_gate_act,
        use_grid_mlp=config.model.use_grid_mlp,
        weight_init=config.model.weight_init,
        norm_type=config.model.norm_type,
    )
    model = EquiformerConsistencyModel(equiformer, config)
    model = model.to(device)
    model.eval()

    # Test 1: shape
    t_mid = torch.full((n_res, 1), 0.5)
    print(f'{t_mid.device=}')
    with torch.no_grad():
        out = model(x_t, t_mid, batch_dict)
    assert out.shape == (n_res, 4), '[FAILED: incorrect output shape]'

    # Test 2: boundary condition
    t_one = torch.ones((n_res, 1))
    with torch.no_grad():
        out_boundary = model(x_t, t_one, batch_dict)
    diff = torch.remainder(out_boundary - x_t + math.pi, 2*math.pi) - math.pi
    err = (diff.abs() * chi_mask).sum() / chi_mask.sum().clamp(1)
    print(f"[{'PASS' if err < 0.1 else 'WARN'}] boundary err at t=1: {err:.4f} rad")

    # Test 3: torus range
    active = out[chi_mask == 1]
    print(f"[{'PASS' if (active >= 0).all() and (active <= 2*math.pi).all() else 'FAIL'}] torus range [{active.min():.3f}, {active.max():.3f}]")

    # Test 4: c_skip/c_out
    print("\nt     c_skip   c_out")
    for tv in [0.0, 0.5, 1.0]:
        t = torch.tensor([[tv]])
        print(f"{tv:.1f}   {model.c_skip(t).item():.4f}   {model.c_out(t).item():.4f}")
    

if __name__ == '__main__':
    _test_equiformer_consistency_model()
