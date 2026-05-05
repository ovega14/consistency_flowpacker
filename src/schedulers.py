"""Custom LR schedulers for training."""
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


__all__ = [
    'ShadowScheduler'
]


# NOTE(OV): basically just to avoid NoneType errors when not using a scheduler
class ShadowScheduler(_LRScheduler):
    """
    Dummy scheduler to act as a placeholder when not using a scheduler.

    'Shadows' an optimizer by simply wrapping around it and only recording
    learning rates during training.

    Args:
        optimizer: Optimizer to shadow during training
    """
    def __init__(self, optimizer: Optimizer, *args, **kwargs):
        super().__init__(optimizer)

    def step(self, *args, **kwargs) -> None:
        pass

    def get_last_lr(self) -> list[float]:
        return [self.optimizer.param_groups[0]['lr']]
