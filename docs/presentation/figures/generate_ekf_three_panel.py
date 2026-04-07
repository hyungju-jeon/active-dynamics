from __future__ import annotations

"""Backward-compatible wrapper for the EKF three-panel figure generator."""

import importlib.util
from pathlib import Path


_IMPL_PATH = Path(__file__).with_name("ekf") / "generate_ekf_three_panel.py"
_SPEC = importlib.util.spec_from_file_location(f"{__name__}._impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load EKF figure generator from {_IMPL_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

__all__ = getattr(
    _MODULE,
    "__all__",
    [name for name in dir(_MODULE) if not name.startswith("_")],
)

globals().update({name: getattr(_MODULE, name) for name in __all__})
