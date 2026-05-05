"""Various helper functions."""
import math


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
