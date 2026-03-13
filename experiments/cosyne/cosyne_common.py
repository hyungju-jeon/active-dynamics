from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def parse_csv_ints(raw: str | None) -> list[int]:
    return [int(item) for item in parse_csv_list(raw)]


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)


def resolve_artifact_path(
    run_dir: str | Path,
    metadata: Mapping[str, Any],
    *,
    key: str,
    fallback_name: str,
) -> Path:
    base_dir = Path(run_dir)
    raw = metadata.get(key)
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path
    return base_dir / fallback_name


def find_nested_metadata_paths(
    root_dir: str | Path,
    *,
    metadata_filename: str = "run_metadata.json",
    nested_pattern: str = "repeat_*",
    include_root: bool = True,
) -> list[Path]:
    base = Path(root_dir)
    paths = sorted(base.glob(f"{nested_pattern}/{metadata_filename}"))
    root_path = base / metadata_filename
    if include_root and root_path.exists():
        paths.append(root_path)
    return paths


def nested_get(
    payload: Mapping[str, Any],
    key_path: Sequence[str],
    default: Any = None,
) -> Any:
    current: Any = payload
    for key in key_path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def load_summary_records(summary_path: str | Path) -> list[dict[str, Any]]:
    payload = load_json(summary_path)
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"Expected 'records' list in {summary_path}")
    return [dict(record) for record in records]


def select_best_record(
    records: Sequence[Mapping[str, Any]],
    *,
    filters: Mapping[str, Any | None] | None = None,
    default_filter: tuple[str, Any] | None = None,
    score_key_path: Sequence[str] = ("post_probe_eval", "rollout_mse"),
) -> dict[str, Any]:
    subset = [dict(record) for record in records]
    filters = dict(filters or {})

    def _matches(record: Mapping[str, Any], key: str, value: Any) -> bool:
        current = record.get(key)
        if value is None:
            return True
        if isinstance(value, int):
            try:
                return int(current) == int(value)
            except Exception:
                return False
        return current == value

    for key, value in filters.items():
        if value is None:
            continue
        subset = [record for record in subset if _matches(record, key, value)]

    if default_filter is not None and all(value is None for value in filters.values()):
        key, value = default_filter
        subset = [record for record in subset if _matches(record, key, value)]

    if not subset:
        raise ValueError("No records matched the requested selection.")

    def _score(record: Mapping[str, Any]) -> float:
        value = nested_get(record, score_key_path)
        numeric = safe_float(value)
        if numeric is None:
            return float("inf")
        return float(numeric)

    return min(subset, key=_score)


def align_trace_length(
    trace: Any,
    num_steps: int,
    *,
    dtype: np.dtype = np.float32,
    empty_error: str,
) -> np.ndarray:
    values = np.asarray(trace, dtype=dtype)
    if values.ndim == 1 and values.size > 0:
        values = values.reshape(1, -1)
    if values.size == 0:
        raise ValueError(empty_error)
    if values.shape[0] < num_steps:
        pad = np.repeat(values[-1:], num_steps - values.shape[0], axis=0)
        values = np.concatenate([values, pad], axis=0)
    if values.shape[0] > num_steps:
        values = values[:num_steps]
    return values


def frame_indices(n_steps: int, stride: int) -> list[int]:
    if n_steps <= 0:
        return [0]
    idxs = list(range(0, n_steps, max(1, int(stride))))
    if idxs[-1] != n_steps - 1:
        idxs.append(n_steps - 1)
    return idxs
