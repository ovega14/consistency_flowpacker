"""Fabrication module for frontend usage."""
import copy
import torch
from torch.optim.lr_scheduler import _LRScheduler

import sys
sys.path.insert(0, '../flowpacker')
from utils.loader import load_config

# Local imports
import schedulers
import consistency
from consistency import EquiformerConsistencyModel


def make_scheduler(sched_type: str, optimizer, **config) -> _LRScheduler:
    """Instantiates a learning rate scheduler."""
    try:  # check PyTorch for native implementation
        scheduler = getattr(torch.optim.lr_scheduler, sched_type)
    except AttributeError:  # check source code for custom implementation
        scheduler = getattr(schedulers, sched_type)
    return scheduler(optimizer, **config)


def make_consistency_models(
    model_type: str,
    device: torch.device,
    config_path: str = '../flowpacker/config/training/vf.yaml',
    ckpt_path: str= '../flowpacker/checkpoints/bc40.pth',    
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """
    Prepares the online and EMA models for consistency distillation.

    If using equiformer implementation, this function builds `EquiformerV2` and
    wraps it in `EquiformerConsistencyModel`, then optionally loads
    FlowPacker checkpoint.

    If instead using local implementation (no equiformer), simply instantiates
    one of the lightweight consistency models based on RCM.

    Args:
        model_type (str): String specifying model type
        device (torch.device): PyTorch device for model residence
        config_path (str): Path to teacher model config

    Returns:
        online_model (Module): Consistency model with parameters to be trained
        ema_model (Module): Consistency model with EMA parameters
    """
    if model_type == 'EquiformerConsistencyModel':
        config = load_config(config_path)  # load teacher model config
        equiformer = EquiformerV2(
            num_layers=config.model.num_layers,
            lmax_list=config.model.lmax_list,
            mmax_list=config.model.mmax_list,
            sphere_channels=config.model.sphere_channels,
            attn_hidden_channels=config.model.attn_hidden_channels,
            num_heads=config.model.num_heads,
            attn_alpha_channels=config.model.attn_alpha_channels,
            attn_value_channels=config.model.attn_value_channels,
            ffn_hidden_channels=config.model.ffn_hidden_channels,
            node_feature_in=config.model.node_feature_in,
            edge_feature_in=config.model.edge_feature_in,
            edge_channels=config.model.edge_channels,
            share_atom_edge_embedding=config.model.share_atom_edge_embedding,
            use_atom_edge_embedding=config.model.use_atom_edge_embedding,
            attn_activation=config.model.attn_activation,
            ffn_activation=config.model.ffn_activation,
            use_gate_act=config.model.use_gate_act,
            use_grid_mlp=config.model.use_grid_mlp,
            weight_init=config.model.weight_init,
            norm_type=config.model.norm_type,
        ).to(device)
        online_model = EquiformerConsistencyModel(equiformer).to(device)
    else:
        online_model = getattr(consistency, model_type)().to(device)

    ema_model = copy.deepcopy(online_model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Initialize from FlowPacker checkpoint
    if ckpt_path and model_type == 'EquiformerConsistencyModel':
        print(f'Loading FlowPacker checkpoint from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = {
            k.replace('module.model.', '') : v for k, v in ckpt['state_dict'].items()
        }
        missing, unexpected = online_model.model.load_state_dict(state_dict, strict=False)
        print(f'  Missing: {len(missing)}, Unexpected: {len(unexpected)}')
        ema_model = copy.deepcopy(online_model)

    return online_model, ema_model
