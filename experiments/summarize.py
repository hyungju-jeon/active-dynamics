#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np
from actdyn.utils.experiment_runtime import read_trace_csv, write_trace_csv
from actdyn.utils.figure_io import sample_sem

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_io import (
        find_nested_metadata_paths,
        get_environment_preset_from_metadata,
        load_json,
        parse_csv_ints,
        parse_csv_list,
        resolve_artifact_path,
        resolve_session_root,
        safe_float,
    )
    from experiment_definitions import get_experiment_spec, list_experiment_ids
else:
    from .experiment_io import (
        find_nested_metadata_paths,
        get_environment_preset_from_metadata,
        load_json,
        parse_csv_ints,
        parse_csv_list,
        resolve_artifact_path,
        resolve_session_root,
        safe_float,
    )
    from .experiment_definitions import get_experiment_spec, list_experiment_ids


TRAJECTORY_R2_THRESHOLDS = (0.90, 0.95, 0.99)


def collect_track_records(
    base_dir: Path,
    exp_id: str,
    seeds: list[int],
    policy_filter: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    exp_spec = get_experiment_spec(exp_id)
    records: list[dict[str, Any]] = []
    missing: list[tuple[str, int]] = []
    track_root = base_dir / exp_id / "track"
    selected_policy_ids = [
        policy_id
        for policy_id in exp_spec.policy_ids
        if policy_filter is None or policy_id in policy_filter
    ]
    for policy_id in selected_policy_ids:
        for seed in seeds:
            seed_dir = track_root / policy_id / f"seed_{seed}"
            if not seed_dir.exists():
                missing.append((policy_id, seed))
                continue
            paths = find_nested_metadata_paths(seed_dir)
            if not paths:
                missing.append((policy_id, seed))
                continue
            for path in paths:
                records.append(
                    {
                        "policy_id": policy_id,
                        "seed": seed,
                        "run_dir": path.parent,
                        "metadata": load_json(path),
                    }
                )
    records.sort(key=lambda rec: (str(rec["policy_id"]), int(rec["seed"]), str(rec["run_dir"])))
    return records, missing


def collect_track_rows(records: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["policy_id"]), int(record["seed"])), []).append(record)
    rows: list[dict[str, Any]] = []
    for (policy_id, seed), runs in grouped.items():
        finals = [safe_float(run["metadata"].get(value_key)) for run in runs]
        runtimes = [safe_float(run["metadata"].get("runtime_sec")) for run in runs]
        trajectory_r2_finals: list[float | None] = []
        for run in runs:
            traj_final = safe_float(run["metadata"].get("trajectory_r2_final"))
            if traj_final is None:
                trace_path = _trace_path(
                    run,
                    metadata_key="trajectory_r2_trace_path",
                    fallback_name="trajectory_r2_trace.csv",
                )
                trace_rows = [] if trace_path is None else read_trace_csv(trace_path)
                if trace_rows:
                    traj_final = safe_float(trace_rows[-1].get("trajectory_r2"))
            trajectory_r2_finals.append(traj_final)
        finals_num = [v for v in finals if v is not None]
        runtimes_num = [v for v in runtimes if v is not None]
        traj_r2_num = [v for v in trajectory_r2_finals if v is not None]
        rows.append(
            {
                "policy_id": policy_id,
                "seed": seed,
                "n_repeats": len(runs),
                "status": (
                    "completed"
                    if all(run["metadata"].get("status") == "completed" for run in runs)
                    else "partial"
                ),
                "value_final_mean": float(np.mean(finals_num)) if finals_num else None,
                "trajectory_r2_final_mean": float(np.mean(traj_r2_num)) if traj_r2_num else None,
                "runtime_sec_mean": float(np.mean(runtimes_num)) if runtimes_num else None,
            }
        )
    rows.sort(key=lambda row: (str(row["policy_id"]), int(row["seed"])))
    return rows


def _trace_path(record: dict[str, Any], metadata_key: str, fallback_name: str) -> Path | None:
    path = resolve_artifact_path(
        record["run_dir"], record["metadata"], key=metadata_key, fallback_name=fallback_name
    )
    return path if path.exists() else None


def aggregate_custom_trace(
    records: list[dict[str, Any]],
    *,
    metadata_key: str,
    fallback_name: str,
    extract_value: Callable[[dict[str, Any]], float | None],
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for policy_id in sorted({str(r["policy_id"]) for r in records}):
        subgroup = [r for r in records if str(r["policy_id"]) == policy_id]
        by_step: dict[int, dict[str, list[float]]] = {}
        for record in subgroup:
            trace_path = _trace_path(record, metadata_key=metadata_key, fallback_name=fallback_name)
            if trace_path is None:
                continue
            for row in read_trace_csv(trace_path):
                step = safe_float(row.get("step"))
                value = extract_value(row)
                cpu_sec = safe_float(row.get("cpu_time_sec"))
                if step is None or value is None:
                    continue
                step_i = int(step)
                bucket = by_step.setdefault(step_i, {"value": [], "cpu": []})
                bucket["value"].append(value)
                if cpu_sec is not None:
                    bucket["cpu"].append(cpu_sec)
        for step_i in sorted(by_step):
            values = by_step[step_i]["value"]
            cpu_vals = by_step[step_i]["cpu"]
            out_rows.append(
                {
                    "policy_id": policy_id,
                    "step": step_i,
                    "value_mean": float(np.mean(values)),
                    "value_sem": sample_sem(values),
                    "cpu_time_sec_mean": float(np.mean(cpu_vals)) if cpu_vals else None,
                    "n_points": len(values),
                }
            )
    out_rows.sort(key=lambda row: (row["policy_id"], int(row["step"])))
    return out_rows


def aggregate_trace(
    records: list[dict[str, Any]],
    *,
    metadata_key: str,
    fallback_name: str,
    value_col: str,
) -> list[dict[str, Any]]:
    return aggregate_custom_trace(
        records,
        metadata_key=metadata_key,
        fallback_name=fallback_name,
        extract_value=lambda row: safe_float(row.get(value_col)),
    )


def _extract_embedding_vector(row: dict[str, Any]) -> np.ndarray | None:
    values: list[tuple[int, float]] = []
    for key, raw in row.items():
        suffix = key.removeprefix("e")
        if not key.startswith("e") or not suffix.isdigit():
            continue
        value = safe_float(raw)
        if value is None:
            return None
        values.append((int(suffix), float(value)))
    if not values:
        return None
    values.sort(key=lambda item: item[0])
    if values[-1][0] != len(values) - 1:
        return None
    return np.asarray([value for _, value in values], dtype=np.float32)


def _recompute_trajectory_trace_rows(
    record: dict[str, Any],
    *,
    exp_spec: Any,
    env_preset: Any,
    interval: int = 10,
) -> list[dict[str, Any]]:
    emb_trace_path = _trace_path(
        record,
        metadata_key="embedding_estimate_trace_path",
        fallback_name="embedding_estimate_trace.csv",
    )
    if emb_trace_path is None:
        return []
    embedding_rows = read_trace_csv(emb_trace_path)
    if not embedding_rows:
        return []

    metadata = record["metadata"]
    true_embedding = metadata.get("embedding_true")
    final_embedding = metadata.get("embedding_estimate")
    if not isinstance(true_embedding, list) or not isinstance(final_embedding, list):
        return []
    expected_dim = max(len(true_embedding), len(final_embedding))
    if expected_dim <= 0:
        return []

    import torch
    from actdyn.utils.validation import trajectory_r2_vectorfield

    out_rows: list[dict[str, Any]] = []
    for row in embedding_rows:
        step = safe_float(row.get("step"))
        cpu_sec = safe_float(row.get("cpu_time_sec"))
        if step is None:
            continue
        step_i = int(step)
        if step_i % int(interval) != 0:
            continue
        embedding = _extract_embedding_vector(row)
        if embedding is None or embedding.shape[0] < expected_dim:
            return []
        local_rng = np.random.default_rng(int(record["seed"]) * 100_000 + step_i + 137)
        out_rows.append(
            {
                "step": step_i,
                "cpu_time_sec": cpu_sec,
                "trajectory_r2": trajectory_r2_vectorfield(
                    e_est=torch.as_tensor(embedding[:expected_dim], dtype=torch.float32),
                    e_true=torch.as_tensor(true_embedding[:expected_dim], dtype=torch.float32),
                    true_dynamics_type=str(metadata.get("dynamics_type") or env_preset.resolved_dynamics_type()),
                    true_full_params=np.asarray(
                        metadata.get("true_params_full") or env_preset.resolved_true_params(),
                        dtype=np.float32,
                    ),
                    estimator_dynamics_type=str(
                        metadata.get("estimator_dynamics_type")
                        or env_preset.resolved_dynamics_type(estimator=True)
                    ),
                    estimator_full_params=np.asarray(
                        metadata.get("estimator_true_params_full")
                        or env_preset.resolved_true_params(estimator=True),
                        dtype=np.float32,
                    ),
                    true_min_embedding_dim=int(
                        metadata.get("min_embedding_dim") or env_preset.resolved_min_embedding_dim()
                    ),
                    estimator_min_embedding_dim=int(
                        metadata.get("min_embedding_dim") or env_preset.resolved_min_embedding_dim()
                    ),
                    dt=float(env_preset.dt),
                    dynamics_alpha=float(env_preset.dynamics_alpha),
                    horizon=int(
                        metadata.get("trajectory_eval_horizon")
                        or exp_spec.trajectory_eval_horizon
                    ),
                    n_starts=int(
                        metadata.get("trajectory_eval_samples")
                        or exp_spec.trajectory_eval_samples
                    ),
                    rng=local_rng,
                    device="cpu",
                ),
                "traj_eval_horizon": int(
                    metadata.get("trajectory_eval_horizon") or exp_spec.trajectory_eval_horizon
                ),
                "traj_eval_samples": int(
                    metadata.get("trajectory_eval_samples") or exp_spec.trajectory_eval_samples
                ),
            }
        )
    return out_rows


def aggregate_trajectory_r2_trace(
    records: list[dict[str, Any]],
    *,
    exp_spec: Any,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for policy_id in sorted({str(r["policy_id"]) for r in records}):
        subgroup = [r for r in records if str(r["policy_id"]) == policy_id]
        by_step: dict[int, dict[str, list[float]]] = {}
        for record in subgroup:
            env_preset = get_environment_preset_from_metadata(record["metadata"])
            trace_rows = _recompute_trajectory_trace_rows(
                record,
                exp_spec=exp_spec,
                env_preset=env_preset,
                interval=10,
            )
            if not trace_rows:
                trace_path = _trace_path(
                    record,
                    metadata_key="trajectory_r2_trace_path",
                    fallback_name="trajectory_r2_trace.csv",
                )
                trace_rows = [] if trace_path is None else read_trace_csv(trace_path)
            for row in trace_rows:
                step = safe_float(row.get("step"))
                value = safe_float(row.get("trajectory_r2"))
                cpu_sec = safe_float(row.get("cpu_time_sec"))
                if step is None or value is None:
                    continue
                step_i = int(step)
                bucket = by_step.setdefault(step_i, {"value": [], "cpu": []})
                bucket["value"].append(value)
                if cpu_sec is not None:
                    bucket["cpu"].append(cpu_sec)
        for step_i in sorted(by_step):
            values = by_step[step_i]["value"]
            cpu_vals = by_step[step_i]["cpu"]
            out_rows.append(
                {
                    "policy_id": policy_id,
                    "step": step_i,
                    "value_mean": float(np.mean(values)),
                    "value_sem": sample_sem(values),
                    "cpu_time_sec_mean": float(np.mean(cpu_vals)) if cpu_vals else None,
                    "n_points": len(values),
                }
            )
    out_rows.sort(key=lambda row: (row["policy_id"], int(row["step"])))
    return out_rows


def _extract_parameter_covariance_trace(row: dict[str, Any]) -> float | None:
    diag_values: list[float] = []
    for key, raw in row.items():
        suffix = key.removeprefix("cov_diag")
        if not key.startswith("cov_diag") or not suffix.isdigit():
            continue
        value = safe_float(raw)
        if value is not None:
            diag_values.append(value)
    if diag_values:
        return float(np.sum(np.asarray(diag_values, dtype=np.float64)))
    return safe_float(row.get("cov_diag_mean"))


def _write_curve_csv(path: Path, rows: list[dict[str, Any]], value_name: str) -> None:
    payloads = [
        {
            "policy_id": row["policy_id"],
            "step": row["step"],
            value_name: row["value_mean"],
            "value_sem": row["value_sem"],
            "cpu_time_sec_mean": row["cpu_time_sec_mean"],
            "n_points": row["n_points"],
        }
        for row in rows
    ]
    write_trace_csv(
        path,
        payloads,
        ["policy_id", "step", value_name, "value_sem", "cpu_time_sec_mean", "n_points"],
    )


def _threshold_suffix(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "p")


def _first_threshold_crossing(
    series: list[dict[str, Any]],
    threshold: float,
) -> dict[str, Any] | None:
    for row in sorted(series, key=lambda r: int(float(r["step"]))):
        value = safe_float(row.get("value_mean"))
        if value is None or not np.isfinite(value) or value < threshold:
            continue
        return row
    return None


def summarize_trajectory_r2_thresholds(
    traj_rows: list[dict[str, Any]],
    thresholds: Sequence[float] = TRAJECTORY_R2_THRESHOLDS,
) -> list[dict[str, Any]]:
    policy_ids = sorted({str(row["policy_id"]) for row in traj_rows})
    out_rows: list[dict[str, Any]] = []
    for policy_id in policy_ids:
        series = [row for row in traj_rows if str(row["policy_id"]) == policy_id]
        payload: dict[str, Any] = {"policy_id": policy_id}
        for threshold in thresholds:
            suffix = _threshold_suffix(float(threshold))
            crossing = _first_threshold_crossing(series, float(threshold))
            if crossing is None:
                payload[f"step_to_r2_{suffix}"] = None
                payload[f"cpu_time_sec_to_r2_{suffix}"] = None
                payload[f"r2_at_{suffix}"] = None
                payload[f"n_points_at_{suffix}"] = None
                continue
            payload[f"step_to_r2_{suffix}"] = int(float(crossing["step"]))
            payload[f"cpu_time_sec_to_r2_{suffix}"] = crossing.get("cpu_time_sec_mean")
            payload[f"r2_at_{suffix}"] = crossing.get("value_mean")
            payload[f"n_points_at_{suffix}"] = crossing.get("n_points")
        out_rows.append(payload)
    out_rows.sort(key=lambda row: str(row["policy_id"]))
    return out_rows


def _write_trajectory_r2_threshold_csv(
    path: Path,
    rows: list[dict[str, Any]],
    thresholds: Sequence[float] = TRAJECTORY_R2_THRESHOLDS,
) -> None:
    fields = ["policy_id"]
    for threshold in thresholds:
        suffix = _threshold_suffix(float(threshold))
        fields.extend(
            [
                f"step_to_r2_{suffix}",
                f"cpu_time_sec_to_r2_{suffix}",
                f"r2_at_{suffix}",
                f"n_points_at_{suffix}",
            ]
        )
    write_trace_csv(path, rows, fields)


def _format_markdown_cell(value: Any, *, digits: int | None = None) -> str:
    parsed = safe_float(value)
    if parsed is None or not np.isfinite(parsed):
        return "--"
    if digits is None:
        return str(int(parsed))
    return f"{parsed:.{digits}f}"


def _write_markdown(
    path: Path,
    exp_id: str,
    rows: list[dict[str, Any]],
    missing: list[tuple[str, int]],
    value_label: str,
    r2_threshold_rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {exp_id} Summary",
        "",
        "## Matrix Coverage",
        "",
        f"- Track rows: {len(rows)}",
        f"- Missing combinations: {len(missing)}",
    ]
    for policy_id, seed in missing:
        lines.append(f"  - `{policy_id}/seed_{seed}`")
    lines.extend(
        [
            "",
            "## Final Metrics by Policy",
            "",
            f"| policy_id | final_{value_label.lower().replace(' ', '_')} | final_trajectory_r2 |",
            "| --- | ---: | ---: |",
        ]
    )
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        policy_bucket = grouped.setdefault(
            str(row["policy_id"]), {"value": [], "trajectory_r2": []}
        )
        policy_bucket["value"].append(
            float(row["value_final_mean"]) if row["value_final_mean"] is not None else np.nan
        )
        policy_bucket["trajectory_r2"].append(
            float(row["trajectory_r2_final_mean"])
            if row.get("trajectory_r2_final_mean") is not None
            else np.nan
        )
    for policy_id in sorted(grouped):
        value_nums = [v for v in grouped[policy_id]["value"] if np.isfinite(v)]
        traj_nums = [v for v in grouped[policy_id]["trajectory_r2"] if np.isfinite(v)]
        lines.append(
            "| "
            + policy_id
            + f" | {float(np.mean(value_nums)) if value_nums else np.nan:.6f}"
            + f" | {float(np.mean(traj_nums)) if traj_nums else np.nan:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory R2 Thresholds",
            "",
            (
                "First sampled environment step and mean CPU time where mean trajectory R2 "
                "reaches each threshold."
            ),
            "",
        ]
    )
    header_cells = ["policy_id"]
    separator_cells = ["---"]
    for threshold in TRAJECTORY_R2_THRESHOLDS:
        label = f"{threshold:.2f}"
        header_cells.extend([f"step_to_R2_{label}", f"cpu_sec_to_R2_{label}"])
        separator_cells.extend(["---:", "---:"])
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(separator_cells) + " |")
    for row in r2_threshold_rows:
        row_cells = [str(row["policy_id"])]
        for threshold in TRAJECTORY_R2_THRESHOLDS:
            suffix = _threshold_suffix(float(threshold))
            row_cells.extend(
                [
                    _format_markdown_cell(row.get(f"step_to_r2_{suffix}")),
                    _format_markdown_cell(row.get(f"cpu_time_sec_to_r2_{suffix}"), digits=2),
                ]
            )
        lines.append("| " + " | ".join(row_cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize COSYNE v2 experiments")
    parser.add_argument("--base-dir", type=str, default="results/cosyne")
    parser.add_argument("--exp-id", choices=list_experiment_ids(), required=True)
    parser.add_argument("--summary-dir", type=str, default=None)
    parser.add_argument("--policy-ids", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0,10,20,30")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    exp_spec = get_experiment_spec(str(args.exp_id))
    base_dir = resolve_session_root(Path(args.base_dir), create=False, exp_ids=[exp_spec.exp_id])
    summary_dir = (
        Path(args.summary_dir) if args.summary_dir else base_dir / exp_spec.exp_id / "summary"
    )
    seeds = parse_csv_ints(args.seeds) or [0, 10, 20, 30]
    policy_filter = set(parse_csv_list(args.policy_ids)) or None
    if policy_filter is not None:
        unknown_policy_ids = sorted(policy_filter - set(exp_spec.policy_ids))
        if unknown_policy_ids:
            parser.error(
                f"Unknown policy ids for {exp_spec.exp_id}: {', '.join(unknown_policy_ids)}"
            )
    records, missing = collect_track_records(
        base_dir, exp_spec.exp_id, seeds, policy_filter=policy_filter
    )
    value_key = "embedding_error_final"
    trace_key = "parameter_error_trace_path"
    trace_name = "parameter_error_trace.csv"
    trace_col = "parameter_error"
    value_prefix = "parameter_error"
    value_label = "Parameter Error"
    rows = collect_track_rows(records, value_key=value_key)
    trace_rows = aggregate_trace(
        records, metadata_key=trace_key, fallback_name=trace_name, value_col=trace_col
    )
    traj_rows = aggregate_trajectory_r2_trace(records, exp_spec=exp_spec)
    r2_threshold_rows = summarize_trajectory_r2_thresholds(traj_rows)
    cov_rows = aggregate_custom_trace(
        records,
        metadata_key="embedding_estimate_trace_path",
        fallback_name="embedding_estimate_trace.csv",
        extract_value=_extract_parameter_covariance_trace,
    )
    info_rows = aggregate_trace(
        records,
        metadata_key="information_trace_path",
        fallback_name="information_trace.csv",
        value_col="I_z_t",
    )
    write_trace_csv(
        summary_dir / "metrics.csv",
        rows,
        [
            "policy_id",
            "seed",
            "n_repeats",
            "status",
            "value_final_mean",
            "trajectory_r2_final_mean",
            "runtime_sec_mean",
        ],
    )
    _write_curve_csv(
        summary_dir / f"{value_prefix}_over_steps.csv", trace_rows, f"{value_prefix}_mean"
    )
    _write_curve_csv(summary_dir / "trajectory_r2_over_steps.csv", traj_rows, "trajectory_r2_mean")
    _write_trajectory_r2_threshold_csv(
        summary_dir / "trajectory_r2_thresholds.csv", r2_threshold_rows
    )
    _write_curve_csv(
        summary_dir / "parameter_covariance_trace_over_steps.csv",
        cov_rows,
        "parameter_covariance_trace_mean",
    )
    _write_curve_csv(summary_dir / "I_z_t_over_steps.csv", info_rows, "I_z_t_mean")
    _write_markdown(
        summary_dir / "metrics.md",
        exp_spec.exp_id,
        rows,
        missing,
        value_label,
        r2_threshold_rows,
    )
    print(f"Wrote {len(rows)} rows to {summary_dir / 'metrics.csv'}")
    if missing and args.fail_on_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
