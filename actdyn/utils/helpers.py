import numpy as np
import torch


def to_np(x: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array."""
    return x.cpu().detach().numpy()
