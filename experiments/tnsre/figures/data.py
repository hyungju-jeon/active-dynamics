"""Shared readers/aggregators over suite summary CSVs.

These are the load/prepare half of the figure pipeline: they read
``<suite>/summary/*.csv`` and return plain rows/values. No plotting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from actdyn.utils.experiment_runtime import read_trace_csv, safe_float
from actdyn.utils.figure_io import sample_sem


def r2_threshold_suffix(threshold: float) -> str:
    return f"{float(threshold):.2f}".replace(".", "p")


def metrics_by_policy(suite_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows = read_trace_csv(suite_dir / "summary" / "metrics.csv")
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row.get("status") not in {None, "", "completed"}:
            continue
        grouped.setdefault(str(row.get("policy_id", "")), []).append(row)
    return grouped


def metric_values(suite_dir: Path, policy_id: str, field: str) -> list[float]:
    values: list[float] = []
    for row in metrics_by_policy(suite_dir).get(policy_id, []):
        value = safe_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def metric_mean_sem(
    suite_dir: Path, policy_id: str, field: str
) -> tuple[float | None, float, int]:
    values = metric_values(suite_dir, policy_id, field)
    if not values:
        return None, 0.0, 0
    return float(np.mean(values)), sample_sem(values), len(values)


def curve_rows(
    suite_dir: Path, name: str, value_col: str
) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for row in read_trace_csv(suite_dir / "summary" / name):
        policy_id = str(row.get("policy_id", ""))
        step = safe_float(row.get("step"))
        value = safe_float(row.get(value_col))
        sem = safe_float(row.get("value_sem"))
        if not policy_id or step is None or value is None:
            continue
        grouped.setdefault(policy_id, []).append(
            {"step": step, "value": value, "sem": 0.0 if sem is None else sem}
        )
    for policy_rows in grouped.values():
        policy_rows.sort(key=lambda row: row["step"])
    return grouped


def r2_threshold_step(
    suite_dir: Path, policy_id: str, threshold: float = 0.90
) -> float | None:
    suffix = r2_threshold_suffix(threshold)
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv"):
        if str(row.get("policy_id", "")) != policy_id:
            continue
        return safe_float(row.get(f"step_to_r2_{suffix}"))
    return None


def r2_threshold_times(
    suite_dir: Path,
    policy_id: str,
    threshold: float,
) -> tuple[float | None, float | None, float | None]:
    suffix = r2_threshold_suffix(threshold)
    for row in read_trace_csv(suite_dir / "summary" / "trajectory_r2_thresholds.csv"):
        if str(row.get("policy_id", "")) != policy_id:
            continue
        return (
            safe_float(row.get(f"step_to_r2_{suffix}")),
            safe_float(row.get(f"cpu_time_sec_to_r2_{suffix}")),
            safe_float(row.get(f"r2_at_{suffix}")),
        )
    return None, None, None
