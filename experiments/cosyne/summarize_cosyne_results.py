#!/usr/bin/env python3
"""Aggregate Cosyne parameter-identification run metadata into CSV/Markdown/figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_EXP_IDS = ["active_short", "active_long", "RND", "random"]
DEFAULT_SEEDS = [0, 10, 20]
DEFAULT_MODEL_TAGS = ["updated"]


def _parse_csv_list(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",")]
    return [item for item in values if item]


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(raw)]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_repeat_metadata_paths(seed_dir: Path) -> list[Path]:
    paths = sorted(seed_dir.glob("repeat_*/run_metadata.json"))
    if (seed_dir / "run_metadata.json").exists():
        paths.append(seed_dir / "run_metadata.json")
    return paths


def collect_track_records(
    base_dir: Path,
    exp_ids: list[str],
    seeds: list[int],
    model_tags: list[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, str, int]]]:
    records: list[dict[str, Any]] = []
    missing: list[tuple[str, str, int]] = []

    for model_tag in model_tags:
        for exp_id in exp_ids:
            for seed in seeds:
                seed_dir = base_dir / "tracks" / model_tag / exp_id / f"seed_{seed}"
                if not seed_dir.exists():
                    missing.append((model_tag, exp_id, seed))
                    continue

                run_metadata_paths = _find_repeat_metadata_paths(seed_dir)
                if not run_metadata_paths:
                    missing.append((model_tag, exp_id, seed))
                    continue

                for path in run_metadata_paths:
                    records.append(
                        {
                            "model_tag": model_tag,
                            "exp_id": exp_id,
                            "seed": seed,
                            "run_dir": path.parent,
                            "metadata": _load_json(path),
                        }
                    )

    records.sort(
        key=lambda rec: (
            str(rec["model_tag"]),
            str(rec["exp_id"]),
            int(rec["seed"]),
            str(rec["run_dir"]),
        )
    )
    return records, missing


def collect_track_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["model_tag"]), str(record["exp_id"]), int(record["seed"]))
        grouped.setdefault(key, []).append(record["metadata"])

    rows: list[dict[str, Any]] = []
    for (model_tag, exp_id, seed), runs in grouped.items():
        final_errors = [
            _safe_float(run.get("embedding_error_final", run.get("latent_error_final")))
            for run in runs
        ]
        mean_errors = [
            _safe_float(run.get("embedding_error_mean", run.get("latent_error_mean")))
            for run in runs
        ]
        steps = [_safe_float(run.get("rollout_steps")) for run in runs]
        runtime = [_safe_float(run.get("runtime_sec")) for run in runs]

        final_errors_num = [v for v in final_errors if v is not None]
        mean_errors_num = [v for v in mean_errors if v is not None]
        steps_num = [v for v in steps if v is not None]
        runtime_num = [v for v in runtime if v is not None]

        nan_any = any(bool(run.get("nan_detected", False)) for run in runs)
        status = "completed" if all(run.get("status") == "completed" for run in runs) else "partial"

        rows.append(
            {
                "model_tag": model_tag,
                "exp_id": exp_id,
                "seed": seed,
                "n_repeats": len(runs),
                "status": status,
                "nan_any": nan_any,
                "embedding_error_final_mean": (
                    float(np.mean(final_errors_num)) if final_errors_num else None
                ),
                "embedding_error_mean_mean": (
                    float(np.mean(mean_errors_num)) if mean_errors_num else None
                ),
                "rollout_steps_mean": float(np.mean(steps_num)) if steps_num else None,
                "runtime_sec_mean": float(np.mean(runtime_num)) if runtime_num else None,
            }
        )

    rows.sort(key=lambda item: (str(item["model_tag"]), str(item["exp_id"]), int(item["seed"])))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model_tag",
        "exp_id",
        "seed",
        "n_repeats",
        "status",
        "nan_any",
        "embedding_error_final_mean",
        "embedding_error_mean_mean",
        "rollout_steps_mean",
        "runtime_sec_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _trace_path(record: dict[str, Any], metadata_key: str, fallback_name: str) -> Path | None:
    metadata = record["metadata"]
    raw = metadata.get(metadata_key)
    if isinstance(raw, str) and raw.strip():
        path = Path(raw)
        if not path.is_absolute():
            path = (record["run_dir"] / path).resolve()
        return path
    fallback = record["run_dir"] / fallback_name
    return fallback if fallback.exists() else None


def _read_trace_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _aggregate_trace(
    records: list[dict[str, Any]],
    *,
    metadata_key: str,
    fallback_name: str,
    value_col: str,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    group_keys = sorted({(str(r["model_tag"]), str(r["exp_id"])) for r in records})

    for model_tag, exp_id in group_keys:
        subgroup = [
            r for r in records if str(r["model_tag"]) == model_tag and str(r["exp_id"]) == exp_id
        ]
        by_step: dict[int, dict[str, list[float]]] = {}

        for record in subgroup:
            trace_path = _trace_path(record, metadata_key=metadata_key, fallback_name=fallback_name)
            if trace_path is None or not trace_path.exists():
                continue

            for row in _read_trace_csv(trace_path):
                step = _safe_float(row.get("step"))
                value = _safe_float(row.get(value_col))
                cpu_sec = _safe_float(row.get("cpu_time_sec"))
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
                    "model_tag": model_tag,
                    "exp_id": exp_id,
                    "step": step_i,
                    "value_mean": float(np.mean(values)),
                    "value_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "cpu_time_sec_mean": float(np.mean(cpu_vals)) if cpu_vals else None,
                    "n_points": len(values),
                }
            )

    out_rows.sort(key=lambda row: (row["model_tag"], row["exp_id"], int(row["step"])))
    return out_rows


def _reconstruct_observation_params(
    seed: int,
    *,
    d_obs: int = 50,
    d_latent: int = 2,
    mean_firing: float = 50.0,
    max_firing_rate: float = 100.0,
    state_range_for_cap: float = 5.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None

    with torch.no_grad():
        torch.manual_seed(int(seed))
        layer = nn.Linear(d_latent, d_obs, bias=True)
        C = layer.weight.detach().clone()
        C[:, 0] = torch.abs(C[:, 0])
        C[:, 1] = C[:, 1] * 2.0

        mean_log_rate = torch.log(torch.full((d_obs,), float(mean_firing)))
        max_log_rate = torch.log(torch.full((d_obs,), float(max_firing_rate)))

        for _ in range(6):
            c_row_l1 = torch.sum(torch.abs(C), dim=1)
            c_row_l2_sq = torch.sum(C * C, dim=1)
            bias_from_mean = mean_log_rate - 0.5 * c_row_l2_sq
            capped_log_rate = float(state_range_for_cap) * c_row_l1 + bias_from_mean
            if torch.all(capped_log_rate <= max_log_rate):
                break
            safe_den = torch.clamp(float(state_range_for_cap) * c_row_l1, min=1e-8)
            row_scale = torch.clamp((max_log_rate - bias_from_mean) / safe_den, min=0.0, max=1.0)
            C = C * row_scale.unsqueeze(1)

        bias = mean_log_rate - 0.5 * torch.sum(C * C, dim=1)
        return C.cpu().numpy(), bias.cpu().numpy()


def _aggregate_mean_firing_trace(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    group_keys = sorted({(str(r["model_tag"]), str(r["exp_id"])) for r in records})
    obs_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}

    for model_tag, exp_id in group_keys:
        subgroup = [
            r for r in records if str(r["model_tag"]) == model_tag and str(r["exp_id"]) == exp_id
        ]
        by_step: dict[int, dict[str, list[float]]] = {}

        for record in subgroup:
            seed = int(record["seed"])
            if seed not in obs_cache:
                obs_cache[seed] = _reconstruct_observation_params(seed)
            params = obs_cache[seed]
            if params is None:
                continue
            C, bias = params

            trace_path = _trace_path(
                record,
                metadata_key="state_action_trace_path",
                fallback_name="state_action_trace.csv",
            )
            if trace_path is None or not trace_path.exists():
                continue

            for row in _read_trace_csv(trace_path):
                step = _safe_float(row.get("step"))
                x = _safe_float(row.get("true_x"))
                v = _safe_float(row.get("true_v"))
                cpu_sec = _safe_float(row.get("cpu_time_sec"))
                if step is None or x is None or v is None:
                    continue
                z = np.asarray([x, v], dtype=np.float64)
                log_rate = C.dot(z) + bias
                rate_hz = float(np.mean(np.exp(np.clip(log_rate, -40.0, 40.0))))
                step_i = int(step)
                bucket = by_step.setdefault(step_i, {"value": [], "cpu": []})
                bucket["value"].append(rate_hz)
                if cpu_sec is not None:
                    bucket["cpu"].append(cpu_sec)

        for step_i in sorted(by_step):
            values = by_step[step_i]["value"]
            cpu_vals = by_step[step_i]["cpu"]
            out_rows.append(
                {
                    "model_tag": model_tag,
                    "exp_id": exp_id,
                    "step": step_i,
                    "value_mean": float(np.mean(values)),
                    "value_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "cpu_time_sec_mean": float(np.mean(cpu_vals)) if cpu_vals else None,
                    "n_points": len(values),
                }
            )

    out_rows.sort(key=lambda row: (row["model_tag"], row["exp_id"], int(row["step"])))
    return out_rows


def _write_curve_csv(path: Path, rows: list[dict[str, Any]], value_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_tag", "exp_id", "step", value_name, "value_std", "cpu_time_sec_mean", "n_points"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = {
                "model_tag": row["model_tag"],
                "exp_id": row["exp_id"],
                "step": row["step"],
                value_name: row["value_mean"],
                "value_std": row["value_std"],
                "cpu_time_sec_mean": row["cpu_time_sec_mean"],
                "n_points": row["n_points"],
            }
            writer.writerow(payload)


def _aggregate_by_track(rows: list[dict[str, Any]], model_tag: str) -> dict[str, float]:
    out: dict[str, float] = {}
    exp_ids = sorted({str(row["exp_id"]) for row in rows if row["model_tag"] == model_tag})
    for exp_id in exp_ids:
        values = [
            _safe_float(row["embedding_error_final_mean"])
            for row in rows
            if row["model_tag"] == model_tag and row["exp_id"] == exp_id
        ]
        values_num = [v for v in values if v is not None]
        if values_num:
            out[exp_id] = float(np.mean(values_num))
    return out


def _ranking(metric_by_track: dict[str, float]) -> list[str]:
    return [k for k, _ in sorted(metric_by_track.items(), key=lambda item: item[1])]


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    missing: list[tuple[str, str, int]],
    model_tags: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Cosyne Parameter-ID Summary")
    lines.append("")
    lines.append("## Matrix Coverage")
    lines.append("")
    lines.append(f"- Track rows: {len(rows)}")
    lines.append(f"- Missing combinations: {len(missing)}")
    for model_tag, exp_id, seed in missing:
        lines.append(f"  - `{model_tag}/{exp_id}/seed_{seed}`")

    by_tag = {tag: _aggregate_by_track(rows, tag) for tag in model_tags}
    lines.append("")

    lines.append("## Parameter Error by Track (Lower Is Better)")
    lines.append("")
    for model_tag in model_tags:
        lines.append(f"### {model_tag}")
        lines.append("")
        lines.append("| exp_id | final_error_mean |")
        lines.append("| --- | ---: |")
        metrics = by_tag.get(model_tag, {})
        for exp_id in sorted(metrics):
            lines.append(f"| {exp_id} | {metrics[exp_id]:.6f} |")
        lines.append("")

    lines.append("## Track Ranking")
    lines.append("")
    for model_tag in model_tags:
        lines.append(f"- {model_tag} ranking: `{_ranking(by_tag.get(model_tag, {}))}`")

    path.write_text("\n".join(lines), encoding="utf-8")


def collect_ablation_rows(base_dir: Path) -> list[dict[str, Any]]:
    ablation_root = base_dir / "ablation"
    if not ablation_root.exists():
        return []

    rows: list[dict[str, Any]] = []
    for path in ablation_root.rglob("run_metadata.json"):
        metadata = _load_json(path)
        axis = metadata.get("ablation_axis")
        value = _safe_float(metadata.get("ablation_value"))
        err = _safe_float(metadata.get("embedding_error_final"))
        if axis is None or value is None or err is None:
            continue
        rows.append(
            {
                "axis": str(axis),
                "value": float(value),
                "model_tag": str(metadata.get("model_tag", "updated")),
                "exp_id": str(metadata.get("exp_id", "unknown")),
                "seed": metadata.get("seed"),
                "embedding_error_final": err,
            }
        )

    grouped: dict[tuple[str, str, float], list[float]] = {}
    for row in rows:
        key = (row["axis"], row["model_tag"], row["value"])
        grouped.setdefault(key, []).append(float(row["embedding_error_final"]))

    agg_rows: list[dict[str, Any]] = []
    for (axis, model_tag, value), vals in grouped.items():
        agg_rows.append(
            {
                "axis": axis,
                "model_tag": model_tag,
                "value": value,
                "embedding_error_final_mean": float(np.mean(vals)),
                "embedding_error_final_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
        )

    agg_rows.sort(key=lambda row: (row["axis"], row["model_tag"], float(row["value"])))
    return agg_rows


def _write_ablation_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "axis",
        "model_tag",
        "value",
        "embedding_error_final_mean",
        "embedding_error_final_std",
        "n",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_figures(
    figures_dir: Path,
    rows: list[dict[str, Any]],
    param_trace_rows: list[dict[str, Any]],
    traj_trace_rows: list[dict[str, Any]],
    firing_trace_rows: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    exp_ids = sorted({str(row["exp_id"]) for row in rows})
    model_tags = sorted({str(row["model_tag"]) for row in rows})

    if rows:
        x = np.arange(len(exp_ids), dtype=np.float64)
        width = 0.35
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for idx, model_tag in enumerate(model_tags):
            values = []
            stds = []
            for exp_id in exp_ids:
                series = [
                    _safe_float(row["embedding_error_final_mean"])
                    for row in rows
                    if row["model_tag"] == model_tag and row["exp_id"] == exp_id
                ]
                nums = [v for v in series if v is not None]
                values.append(float(np.mean(nums)) if nums else np.nan)
                stds.append(float(np.std(nums, ddof=1)) if len(nums) > 1 else 0.0)
            offset = (idx - (len(model_tags) - 1) / 2.0) * width
            ax.bar(x + offset, values, width=width, yerr=stds, capsize=3, label=model_tag)

        ax.set_xticks(x)
        ax.set_xticklabels(exp_ids, rotation=20)
        ax.set_ylabel("Final Parameter Error (mean ± STD over seeds)")
        ax.set_title("Cosyne: Final Parameter Error by Track")
        ax.legend(loc="best")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(figures_dir / "final_error_by_track.png", dpi=150)
        plt.close(fig)

    if param_trace_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        key_pairs = sorted({(r["model_tag"], r["exp_id"]) for r in param_trace_rows})
        single_tag = len({k[0] for k in key_pairs}) == 1
        for model_tag, exp_id in key_pairs:
            series = [
                r
                for r in param_trace_rows
                if r["model_tag"] == model_tag and r["exp_id"] == exp_id
            ]
            series.sort(key=lambda r: int(r["step"]))
            xs = [int(r["step"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            std = [float(r["value_std"]) for r in series]
            label = exp_id if single_tag else f"{model_tag}:{exp_id}"
            ax.plot(xs, ys, label=label)
            ax.fill_between(
                xs,
                np.asarray(ys) - np.asarray(std),
                np.asarray(ys) + np.asarray(std),
                alpha=0.18,
            )
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("Parameter Error (mean ± STD over seeds)")
        ax.set_title("Parameter Error Over Time Steps (with STD)")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / "parameter_error_over_steps.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for model_tag, exp_id in key_pairs:
            series = [
                r
                for r in param_trace_rows
                if r["model_tag"] == model_tag and r["exp_id"] == exp_id
            ]
            series = [r for r in series if r["cpu_time_sec_mean"] is not None]
            series.sort(key=lambda r: int(r["step"]))
            xs = [float(r["cpu_time_sec_mean"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            std = [float(r["value_std"]) for r in series]
            label = exp_id if single_tag else f"{model_tag}:{exp_id}"
            ax.plot(xs, ys, label=label)
            ax.fill_between(
                xs,
                np.asarray(ys) - np.asarray(std),
                np.asarray(ys) + np.asarray(std),
                alpha=0.18,
            )
        ax.set_xlabel("CPU Time (sec)")
        ax.set_ylabel("Parameter Error (mean ± STD over seeds)")
        ax.set_title("Parameter Error Over CPU Time (with STD)")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / "parameter_error_over_cpu_time.png", dpi=150)
        plt.close(fig)

    if traj_trace_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        key_pairs = sorted({(r["model_tag"], r["exp_id"]) for r in traj_trace_rows})
        single_tag = len({k[0] for k in key_pairs}) == 1
        for model_tag, exp_id in key_pairs:
            series = [
                r
                for r in traj_trace_rows
                if r["model_tag"] == model_tag and r["exp_id"] == exp_id
            ]
            series.sort(key=lambda r: int(r["step"]))
            xs = [int(r["step"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            std = [float(r["value_std"]) for r in series]
            label = exp_id if single_tag else f"{model_tag}:{exp_id}"
            ax.plot(xs, ys, label=label)
            ax.fill_between(
                xs,
                np.asarray(ys) - np.asarray(std),
                np.asarray(ys) + np.asarray(std),
                alpha=0.18,
            )
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("Trajectory R2 (mean ± STD over seeds)")
        ax.set_title("Trajectory R2 Over Time (No-input rollout checks, with STD)")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / "trajectory_r2_over_steps.png", dpi=150)
        plt.close(fig)

    if firing_trace_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        key_pairs = sorted({(r["model_tag"], r["exp_id"]) for r in firing_trace_rows})
        single_tag = len({k[0] for k in key_pairs}) == 1
        for model_tag, exp_id in key_pairs:
            series = [
                r
                for r in firing_trace_rows
                if r["model_tag"] == model_tag and r["exp_id"] == exp_id
            ]
            series.sort(key=lambda r: int(r["step"]))
            xs = [int(r["step"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            std = [float(r["value_std"]) for r in series]
            label = exp_id if single_tag else f"{model_tag}:{exp_id}"
            ax.plot(xs, ys, label=label)
            ax.fill_between(
                xs,
                np.asarray(ys) - np.asarray(std),
                np.asarray(ys) + np.asarray(std),
                alpha=0.18,
            )
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("Mean Firing Rate (Hz, mean ± STD over seeds)")
        ax.set_title("Mean Firing Rate Over Time Steps (with STD)")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figures_dir / "mean_firing_rate_over_steps.png", dpi=150)
        plt.close(fig)

    if ablation_rows:
        axis_cfg = {
            "planning_window": ("Planning Window", "ablation_planning_window.png"),
            "update_frequency": ("Parameter Update Frequency (k_theta)", "ablation_update_frequency.png"),
        }
        for axis, (xlabel, filename) in axis_cfg.items():
            subset = [r for r in ablation_rows if r["axis"] == axis]
            if not subset:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            for model_tag in sorted({r["model_tag"] for r in subset}):
                rows_tag = [r for r in subset if r["model_tag"] == model_tag]
                rows_tag.sort(key=lambda r: float(r["value"]))
                xs = [float(r["value"]) for r in rows_tag]
                ys = [float(r["embedding_error_final_mean"]) for r in rows_tag]
                ax.plot(xs, ys, marker="o", label=model_tag)

            ax.set_xlabel(xlabel)
            ax.set_ylabel("Final Parameter Error (mean)")
            ax.set_title(f"Ablation: {xlabel}")
            ax.grid(alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(figures_dir / filename, dpi=150)
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Cosyne parameter-identification runs")
    parser.add_argument("--base-dir", type=str, default="results/cosyne")
    parser.add_argument("--summary-dir", type=str, default="results/cosyne/summary")
    parser.add_argument("--exp-ids", type=str, default=",".join(DEFAULT_EXP_IDS))
    parser.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--model-tags", type=str, default=",".join(DEFAULT_MODEL_TAGS))
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    base_dir = Path(args.base_dir)
    summary_dir = Path(args.summary_dir)
    exp_ids = _parse_csv_list(args.exp_ids) or list(DEFAULT_EXP_IDS)
    seeds = _parse_csv_ints(args.seeds) or list(DEFAULT_SEEDS)
    model_tags = _parse_csv_list(args.model_tags) or list(DEFAULT_MODEL_TAGS)

    records, missing = collect_track_records(
        base_dir=base_dir,
        exp_ids=exp_ids,
        seeds=seeds,
        model_tags=model_tags,
    )
    rows = collect_track_rows(records)

    param_trace_rows = _aggregate_trace(
        records,
        metadata_key="parameter_error_trace_path",
        fallback_name="parameter_error_trace.csv",
        value_col="parameter_error",
    )
    traj_trace_rows = _aggregate_trace(
        records,
        metadata_key="trajectory_r2_trace_path",
        fallback_name="trajectory_r2_trace.csv",
        value_col="trajectory_r2",
    )
    firing_trace_rows = _aggregate_mean_firing_trace(records)

    ablation_rows = collect_ablation_rows(base_dir)

    _write_csv(summary_dir / "metrics.csv", rows)
    _write_curve_csv(summary_dir / "parameter_error_over_steps.csv", param_trace_rows, "parameter_error_mean")
    _write_curve_csv(summary_dir / "trajectory_r2_over_steps.csv", traj_trace_rows, "trajectory_r2_mean")
    _write_curve_csv(
        summary_dir / "mean_firing_rate_over_steps.csv",
        firing_trace_rows,
        "mean_firing_rate_hz_mean",
    )
    _write_ablation_csv(summary_dir / "ablation_metrics.csv", ablation_rows)
    _write_markdown(summary_dir / "metrics.md", rows, missing, model_tags=model_tags)
    _plot_figures(
        summary_dir / "figures",
        rows=rows,
        param_trace_rows=param_trace_rows,
        traj_trace_rows=traj_trace_rows,
        firing_trace_rows=firing_trace_rows,
        ablation_rows=ablation_rows,
    )

    print(f"Wrote {len(rows)} track rows to {summary_dir / 'metrics.csv'}")
    if missing:
        print(f"Missing matrix entries: {len(missing)}")
        for model_tag, exp_id, seed in missing:
            print(f"- {model_tag}/{exp_id}/seed_{seed}")
        if args.fail_on_missing:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
