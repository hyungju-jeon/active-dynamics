from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def configure_runtime(seed: int = 0, device: str | None = None) -> str:
    """Configure deterministic runtime defaults and resolve device."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def ensure_dir(path: str | Path) -> str:
    """Create directory if needed and return it as a string."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)
