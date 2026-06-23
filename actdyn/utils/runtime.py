from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

import numpy as np
import torch


def configure_runtime(seed: int = 0, device: str | None = None) -> str:
    """Configure deterministic runtime defaults and resolve device."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_threads = int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or 1)
    torch.set_num_threads(max(1, num_threads))
    interop_threads = int(os.environ.get("TORCH_NUM_INTEROP_THREADS") or 1)
    try:
        torch.set_num_interop_threads(max(1, interop_threads))
    except RuntimeError:
        pass

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


def repo_root() -> Path:
    """Return the repository root based on the installed actdyn package layout."""
    return Path(__file__).resolve().parents[2]


def resolve_repo_path(path_like: str | Path) -> Path:
    """Resolve an absolute path or interpret a relative path from the repo root."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (repo_root() / path).resolve()


def utc_now() -> str:
    """Return the current UTC time in ISO-8601 format with a Z suffix."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def current_commit(*, short: bool = True) -> str:
    """Return the current git commit hash for the repository, or 'unknown' on failure."""
    rev = "--short" if short else "HEAD"
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", rev, "HEAD"] if short else ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root()),
            text=True,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"
