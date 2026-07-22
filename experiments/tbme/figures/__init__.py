"""TBME figure package — one module per figure family.

Migration status: families still living in the legacy split modules
(``tbme_figures_summary/_experiment/_assets/_diagnostics``) are being moved
here one family per change. The legacy modules keep their public names as
aliases while any consumer remains.

Dispatch is an explicit table (no decorator registration); imports happen
inside the wrappers so importing this package stays cheap and cycle-free.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def _objective_ablation() -> list[Path]:
    from experiments.tbme.figures import ablation

    return ablation.generate()


FIGURES: dict[str, Callable[[], list[Path]]] = {
    "objective_ablation": _objective_ablation,
}
