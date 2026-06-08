from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


SUPPORTED_FIGURE_FORMATS = frozenset({".pdf", ".png", ".svg"})


def parse_figure_formats(raw: str) -> tuple[str, ...]:
    """Parse comma-separated output formats for experiment figure files."""
    formats: list[str] = []
    for item in str(raw).split(","):
        fmt = item.strip().lower()
        if not fmt:
            continue
        if not fmt.startswith("."):
            fmt = f".{fmt}"
        if fmt not in SUPPORTED_FIGURE_FORMATS:
            expected = ", ".join(sorted(SUPPORTED_FIGURE_FORMATS))
            raise ValueError(f"Unsupported figure format {item!r}. Expected one of: {expected}")
        if fmt not in formats:
            formats.append(fmt)
    return tuple(formats) if formats else (".pdf",)


def sample_sem(values: Sequence[float]) -> float:
    """Return the sample standard error of the mean for one scalar sample set."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size))


def load_plotting(
    output_path: Path,
    *,
    apply_style: Callable[[Any], None] | None = None,
    path_is_file: bool = False,
    use_agg: bool = True,
) -> Any | None:
    """Load Matplotlib lazily and create the figure output directory."""
    if path_is_file:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    try:
        if use_agg:
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
    except Exception:
        return None
    if apply_style is not None:
        apply_style(plt_module)
    return plt_module


def save_figure(
    fig: Any,
    output_path: Path,
    *,
    plt_module: Any | None = None,
    close: bool = True,
    dpi: int | None = None,
) -> Path:
    """Save one Matplotlib figure file and optionally close the figure."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"bbox_inches": "tight", "pad_inches": 0.02}
    if dpi is not None:
        save_kwargs["dpi"] = int(dpi)
    fig.savefig(output_path, **save_kwargs)
    if close:
        if plt_module is None:
            import matplotlib.pyplot as plt_module
        plt_module.close(fig)
    return output_path


def save_figure_formats(
    fig: Any,
    stem_path: Path,
    figure_formats: Sequence[str],
    *,
    plt_module: Any | None = None,
) -> list[Path]:
    """Save one figure stem using each requested validated extension."""
    paths: list[Path] = []
    for fmt in figure_formats:
        dpi = 300 if fmt == ".png" else None
        paths.append(save_figure(fig, stem_path.with_suffix(fmt), close=False, dpi=dpi))
    if plt_module is None:
        import matplotlib.pyplot as plt_module
    plt_module.close(fig)
    return paths
