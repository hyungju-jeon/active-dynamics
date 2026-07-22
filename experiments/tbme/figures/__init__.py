"""TBME figure package — one module per figure family.

Layers: ``theme``/``groups``/``artifacts``/``data``/``records``/``information``
are shared infrastructure; every other module is one figure family following
``prepare() -> FamilyData -> plot builder -> generate()`` with figure and CSV
sidecars produced from the same prepared data. ``cli`` and ``assets`` hold the
experiment-level and manuscript-asset entry points; the documented CLI router
is ``experiments/tbme/generate_figures.py``.

Dispatch is an explicit table (no decorator registration); imports happen
inside the wrappers so importing this package stays cheap and cycle-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def _objective_ablation() -> list[Path]:
    from experiments.tbme.figures import ablation

    return ablation.generate()


def _bottleneck_sweep() -> list[Path]:
    from experiments.tbme.figures import bottleneck

    return bottleneck.generate()


def _mismatch_dose_response() -> list[Path]:
    from experiments.tbme.figures import mismatch

    return mismatch.generate()


FIGURES: dict[str, Callable[[], list[Path]]] = {
    "objective_ablation": _objective_ablation,
    "bottleneck_sweep": _bottleneck_sweep,
    "mismatch_dose_response": _mismatch_dose_response,
}
