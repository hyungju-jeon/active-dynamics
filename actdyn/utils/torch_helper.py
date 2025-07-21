import torch
import numpy as np


def activation_from_str(activation_str: str):
    """Convert a string to a PyTorch activation function."""
    if activation_str is None:
        return None
    if isinstance(activation_str, str):
        activation_str = activation_str.lower()
        if activation_str == "relu":
            return torch.nn.ReLU()
        elif activation_str == "tanh":
            return torch.nn.Tanh()
        elif activation_str == "sigmoid":
            return torch.nn.Sigmoid()
        elif activation_str == "leakyrelu":
            return torch.nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")


def to_np(x: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array."""
    return x.cpu().detach().numpy()
