"""Suite/group resolution for TBME figures.

The GROUPS table itself still lives in ``experiments.tbme.tbme_figures`` (it is
built at import time and mutated by ``--results-dir``); the accessors here read
it dynamically so they always see the current table instead of a stale binding.
They become the real implementation once the last legacy figure module
migrates into this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SuiteSource:
    """One suite feeding a figure: an experiment id, display label, and data dir."""

    exp_id: str
    label: str
    suite_dir: Path
    dose: str | None = None
    family: str | None = None


def _legacy() -> Any:
    from experiments.tbme import tbme_figures

    return tbme_figures


def groups() -> dict[str, list[Any]]:
    return _legacy().GROUPS


def suite_dir(group_name: str, suite_id: str) -> Path:
    return _legacy()._suite_dir(group_name, suite_id)
