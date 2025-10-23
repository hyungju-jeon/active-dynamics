"""
General helper functions and constants
"""

import torch
import numpy as np

eps = 1e-6


# -------------------------------------------------------------
# General Helpers
# -------------------------------------------------------------
def format_list(x):
    fstr = ", ".join([f"{val:.3f}" for val in x.reshape(-1).tolist()])
    return "(" + fstr + ")"


# -------------------------------------------------------------
# Torch Helpers
# -------------------------------------------------------------
def to_np(x: torch.Tensor) -> np.ndarray:
    """Converts a PyTorch tensor to a NumPy array."""
    return x.cpu().detach().numpy()


def safe_cholesky(M, jitter=1e-6, max_tries=5, growth=10.0):
    I = torch.eye(M.size(-1), device=M.device).expand_as(M)
    j = 0.0
    for _ in range(max_tries):
        try:
            return torch.linalg.cholesky(M + j * I)
        except RuntimeError:
            j = jitter if j == 0.0 else j * growth
    return torch.linalg.cholesky(M + j * I)


def symmetrize(M):
    return 0.5 * (M + M.transpose(-1, -2))


def activation_from_str(activation_str: str):
    """Convert a string to a PyTorch activation function.

    DESIGN NOTE: Incomplete activation registry
    ============================================

    Issue: Tests use "leaky_relu" (with underscore) but registry only supports "leakyrelu" (no underscore)
    Impact: Multiple test failures across encoder, decoder, dynamics tests

    Better design:
    1. Support common variations (with/without underscore, camelCase, etc.)
    2. Add comprehensive list of activations used in config.py defaults
    3. Consider using a dict-based registry for easier extension

    Recommended registry:
    ACTIVATION_MAP = {
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid,
        'leakyrelu': torch.nn.LeakyReLU,
        'leaky_relu': torch.nn.LeakyReLU,  # Common alias
        'elu': torch.nn.ELU,
        'gelu': torch.nn.GELU,
        'selu': torch.nn.SELU,
        'prelu': torch.nn.PReLU,
        'swish': torch.nn.SiLU,  # Swish = SiLU
        'silu': torch.nn.SiLU,
    }

    Usage in codebase:
    - config.py mentions "leaky_relu" in comments as option
    - Multiple components (encoder, decoder, dynamics) use this
    """
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
        # DESIGN FIX: Add common alias with underscore
        elif activation_str == "leaky_relu":
            return torch.nn.LeakyReLU()
        elif activation_str == "elu":
            return torch.nn.ELU()
        else:
            raise ValueError(f"Unknown activation function: {activation_str}")
