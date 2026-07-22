"""Run-record discovery and trace loading for TBME figures.

A run record is one (policy, seed) run directory with its parsed metadata;
figures aggregate over records rather than touching directories directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ...experiment_io import find_nested_metadata_paths, load_json
from ..tbme_io import read_xy_trace, trace_path as run_trace_path
from . import theme


@dataclass(frozen=True)
class RunRecord:
    policy_id: str
    seed: int
    run_dir: Path
    metadata: dict[str, Any]


def collect_records(
    suite_dir: Path,
    policy_ids: Sequence[str],
    *,
    max_seeds: int | None = None,
    completed_only: bool = False,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for policy_id in sorted(policy_ids, key=theme.policy_sort_key):
        policy_dir = suite_dir / policy_id
        if not policy_dir.exists():
            continue
        seed_dirs: list[tuple[int, Path]] = []
        for seed_dir in policy_dir.glob("seed_*"):
            suffix = seed_dir.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), seed_dir))
        selected_seed_dirs = sorted(seed_dirs)
        if max_seeds is not None:
            selected_seed_dirs = selected_seed_dirs[: int(max_seeds)]
        for seed, seed_dir in selected_seed_dirs:
            for metadata_path in find_nested_metadata_paths(seed_dir):
                metadata = load_json(metadata_path)
                if completed_only and metadata.get("status") != "completed":
                    continue
                records.append(
                    RunRecord(
                        policy_id=policy_id,
                        seed=seed,
                        run_dir=metadata_path.parent,
                        metadata=metadata,
                    )
                )
    return records


def state_bounds_from_metadata(metadata: dict[str, Any]) -> tuple[float, float]:
    low = np.asarray(metadata.get("state_low", [-5.0, -5.0]), dtype=np.float64)
    high = np.asarray(metadata.get("state_high", [5.0, 5.0]), dtype=np.float64)
    return float(np.min(low)), float(np.max(high))


def record_trace_path(record: RunRecord, metadata_key: str, fallback_name: str) -> Path:
    return run_trace_path(record.run_dir, record.metadata, metadata_key, fallback_name)


def load_xy_trace(record: RunRecord) -> np.ndarray:
    path = record_trace_path(record, "state_action_trace_path", "state_action_trace.csv")
    return read_xy_trace(path)
