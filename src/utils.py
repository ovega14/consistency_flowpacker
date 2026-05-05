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
