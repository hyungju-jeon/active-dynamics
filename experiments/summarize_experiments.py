#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from experiment_common import (
        expected_loglinear_rate_hz,
        find_nested_metadata_paths,
        get_environment_preset_from_metadata,
        load_json,
        parse_csv_ints,
        parse_csv_list,
        reconstruct_loglinear_rate_model,
        resolve_artifact_path,
        resolve_session_root,
        safe_float,
    )
    from experiment_specs import get_experiment_spec, list_experiment_ids
    from experiments.cosyne.planar_systems import (
        get_planar_system_spec,
        has_planar_system_spec,
        residual_torch,
    )
else:
    from .experiment_common import (
        expected_loglinear_rate_hz,
        find_nested_metadata_paths,
        get_environment_preset_from_metadata,
        load_json,
        parse_csv_ints,
        parse_csv_list,
        reconstruct_loglinear_rate_model,
        resolve_artifact_path,
        resolve_session_root,
        safe_float,
    )
    from .experiment_specs import get_experiment_spec, list_experiment_ids
    from .cosyne.planar_systems import (
        get_planar_system_spec,
        has_planar_system_spec,
        residual_torch,
    )
from actdyn.utils.visualize import (
    PlanarResidualDynamics,
    decorate_phase_space_axis,
    plot_vector_field,
)


SUPPORTED_FIGURE_FORMATS = frozenset({".pdf", ".png", ".svg"})


def collect_track_records(
    base_dir: Path, exp_id: str, seeds: list[int]
) -> tuple[list[dict[str, Any]], list[tuple[str, int]]]:
    exp_spec = get_experiment_spec(exp_id)
    records: list[dict[str, Any]] = []
    missing: list[tuple[str, int]] = []
    track_root = base_dir / exp_id / "track"
    for policy_id in exp_spec.policy_ids:
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
                trace_rows = [] if trace_path is None else _read_trace_csv(trace_path)
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


def _read_trace_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
            for row in _read_trace_csv(trace_path):
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
                    "value_sem": _sample_sem(values),
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


def _sample_sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    _write_csv(
        path,
        payloads,
        ["policy_id", "step", value_name, "value_sem", "cpu_time_sec_mean", "n_points"],
    )


def _write_markdown(
    path: Path,
    exp_id: str,
    rows: list[dict[str, Any]],
    missing: list[tuple[str, int]],
    value_label: str,
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_figure_formats(raw: str) -> tuple[str, ...]:
    formats: list[str] = []
    for item in parse_csv_list(raw):
        fmt = item.lower()
        if not fmt.startswith("."):
            fmt = f".{fmt}"
        if fmt not in SUPPORTED_FIGURE_FORMATS:
            expected = ", ".join(sorted(SUPPORTED_FIGURE_FORMATS))
            raise ValueError(f"Unsupported figure format {item!r}. Expected one of: {expected}")
        if fmt not in formats:
            formats.append(fmt)
    return tuple(formats) if formats else (".pdf",)


def _save_figure(fig: Any, stem_path: Path, figure_formats: Sequence[str]) -> None:
    for fmt in figure_formats:
        save_kwargs = {"dpi": 150} if fmt == ".png" else {}
        fig.savefig(stem_path.with_suffix(fmt), **save_kwargs)


def _plot_curves(
    figures_dir: Path,
    *,
    rows: list[dict[str, Any]],
    trace_rows: list[dict[str, Any]],
    cov_rows: list[dict[str, Any]],
    traj_rows: list[dict[str, Any]],
    value_label: str,
    value_prefix: str,
    figure_formats: Sequence[str],
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    if rows:
        policy_ids = sorted({str(row["policy_id"]) for row in rows})
        means = []
        sems = []
        for policy_id in policy_ids:
            vals = [
                safe_float(row["value_final_mean"]) for row in rows if row["policy_id"] == policy_id
            ]
            nums = [v for v in vals if v is not None]
            means.append(float(np.mean(nums)) if nums else np.nan)
            sems.append(_sample_sem(nums))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        x = np.arange(len(policy_ids), dtype=np.float64)
        ax.bar(x, means, yerr=sems, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(policy_ids, rotation=20)
        ax.set_ylabel(f"Final {value_label} (mean ± SEM over seeds)")
        ax.set_title(f"{value_label} by Policy")
        ax.grid(alpha=0.2, axis="y")
        fig.tight_layout()
        _save_figure(fig, figures_dir / f"final_{value_prefix}_by_policy", figure_formats)
        plt.close(fig)
    if trace_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for policy_id in sorted({str(r["policy_id"]) for r in trace_rows}):
            series = [r for r in trace_rows if r["policy_id"] == policy_id]
            series.sort(key=lambda r: int(r["step"]))
            xs = [int(r["step"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            sem = [float(r["value_sem"]) for r in series]
            ax.plot(xs, ys, label=policy_id)
            ax.fill_between(
                xs, np.asarray(ys) - np.asarray(sem), np.asarray(ys) + np.asarray(sem), alpha=0.18
            )
        ax.set_xlabel("Environment Step")
        ax.set_ylabel(f"{value_label} (mean ± SEM)")
        ax.set_title(f"{value_label} Over Steps")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, figures_dir / f"{value_prefix}_over_steps", figure_formats)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for policy_id in sorted({str(r["policy_id"]) for r in trace_rows}):
            series = [
                r
                for r in trace_rows
                if r["policy_id"] == policy_id and r["cpu_time_sec_mean"] is not None
            ]
            series.sort(key=lambda r: int(r["step"]))
            xs = [float(r["cpu_time_sec_mean"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            sem = [float(r["value_sem"]) for r in series]
            ax.plot(xs, ys, label=policy_id)
            ax.fill_between(
                xs, np.asarray(ys) - np.asarray(sem), np.asarray(ys) + np.asarray(sem), alpha=0.18
            )
        ax.set_xlabel("CPU Time (sec)")
        ax.set_ylabel(f"{value_label} (mean ± SEM)")
        ax.set_title(f"{value_label} Over CPU Time")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, figures_dir / f"{value_prefix}_over_cpu_time", figure_formats)
        plt.close(fig)
    if traj_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for policy_id in sorted({str(r["policy_id"]) for r in traj_rows}):
            series = [r for r in traj_rows if r["policy_id"] == policy_id]
            series.sort(key=lambda r: int(r["step"]))
            xs = [int(r["step"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            sem = [float(r["value_sem"]) for r in series]
            ax.plot(xs, ys, label=policy_id)
            ax.fill_between(
                xs, np.asarray(ys) - np.asarray(sem), np.asarray(ys) + np.asarray(sem), alpha=0.18
            )
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("Trajectory R2 (mean ± SEM)")
        ax.set_title("Trajectory R2 Over Steps")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, figures_dir / "trajectory_r2_over_steps", figure_formats)
        plt.close(fig)
    if cov_rows:
        fig, ax = plt.subplots(figsize=(9.5, 5.0))
        for policy_id in sorted({str(r["policy_id"]) for r in cov_rows}):
            series = [r for r in cov_rows if r["policy_id"] == policy_id]
            series.sort(key=lambda r: int(r["step"]))
            xs = [int(r["step"]) for r in series]
            ys = [float(r["value_mean"]) for r in series]
            sem = [float(r["value_sem"]) for r in series]
            ax.plot(xs, ys, label=policy_id)
            ax.fill_between(
                xs, np.asarray(ys) - np.asarray(sem), np.asarray(ys) + np.asarray(sem), alpha=0.18
            )
        ax.set_xlabel("Environment Step")
        ax.set_ylabel("Trace of Parameter Covariance (mean ± SEM)")
        ax.set_title("Trace of Parameter Covariance Over Steps")
        ax.grid(alpha=0.2)
        ax.legend(loc="best")
        fig.tight_layout()
        _save_figure(fig, figures_dir / "parameter_covariance_trace_over_steps", figure_formats)
        plt.close(fig)


def _reference_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    return min(
        records,
        key=lambda rec: (
            int(rec["seed"]),
            str(rec["policy_id"]),
            str(rec["run_dir"]),
        ),
    )


def _seed_reference_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    refs: dict[int, dict[str, Any]] = {}
    for record in sorted(
        records,
        key=lambda rec: (int(rec["seed"]), str(rec["policy_id"]), str(rec["run_dir"])),
    ):
        refs.setdefault(int(record["seed"]), record)
    return [refs[seed] for seed in sorted(refs)]


def _build_policy_step_matrix(
    rows: list[dict[str, Any]],
) -> tuple[list[str], np.ndarray, np.ndarray] | None:
    if not rows:
        return None
    policy_ids = sorted({str(row["policy_id"]) for row in rows})
    steps = np.asarray(sorted({int(row["step"]) for row in rows}), dtype=int)
    matrix = np.full((len(policy_ids), len(steps)), np.nan, dtype=np.float32)
    policy_to_idx = {policy_id: idx for idx, policy_id in enumerate(policy_ids)}
    step_to_idx = {int(step): idx for idx, step in enumerate(steps.tolist())}
    for row in rows:
        policy_idx = policy_to_idx[str(row["policy_id"])]
        step_idx = step_to_idx[int(row["step"])]
        matrix[policy_idx, step_idx] = float(row["value_mean"])
    return policy_ids, steps, matrix


def _plot_neuron_tuning_curve_colormap(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
) -> None:
    seed_refs = _seed_reference_records(records)
    if not seed_refs:
        return
    metadata = dict(seed_refs[0]["metadata"])
    if not has_planar_system_spec(str(metadata.get("system_id", "")).strip()):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    env_preset = get_environment_preset_from_metadata(dict(seed_refs[0]["metadata"]))
    grid_lim = float(env_preset.x_range)
    n_grid = 121
    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)

    maps: list[np.ndarray] = []
    for ref in seed_refs:
        metadata = dict(ref["metadata"])
        weights, bias, _dt = reconstruct_loglinear_rate_model(metadata)
        rate_hz = expected_loglinear_rate_hz(latent, weights=weights, bias=bias)
        maps.append(np.sum(rate_hz, axis=1).reshape(n_grid, n_grid))
    heat = np.mean(np.stack(maps, axis=0), axis=0)

    finite = heat[np.isfinite(heat)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = ax.imshow(
        heat,
        aspect="equal",
        origin="lower",
        extent=[-grid_lim, grid_lim, -grid_lim, grid_lim],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Total firing rate (Hz)")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Total Firing Rate Colormap (mean over {len(seed_refs)} seed(s))")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "neuron_tuning_curve_colormap", figure_formats)
    plt.close(fig)


def _plot_information_colormap(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
) -> None:
    seed_refs = _seed_reference_records(records)
    if not seed_refs:
        return
    metadata = dict(seed_refs[0]["metadata"])
    if not has_planar_system_spec(str(metadata.get("system_id", "")).strip()):
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    env_preset = get_environment_preset_from_metadata(dict(seed_refs[0]["metadata"]))
    grid_lim = float(env_preset.x_range)
    n_grid = 121
    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    latent = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    eye = np.eye(2, dtype=np.float32)

    maps: list[np.ndarray] = []
    for ref in seed_refs:
        metadata = dict(ref["metadata"])
        weights, bias, dt = reconstruct_loglinear_rate_model(metadata)
        rate_hz = expected_loglinear_rate_hz(latent, weights=weights, bias=bias)
        mean_counts = np.clip(rate_hz * float(dt), 1e-8, 1e8)
        info_mats = np.einsum("nd,di,dj->nij", mean_counts, weights, weights, optimize=True)
        info_mats = info_mats + 1e-9 * eye[None, :, :]
        sign, logabsdet = np.linalg.slogdet(info_mats)
        logdet = np.where(sign > 0.0, logabsdet, np.nan).reshape(n_grid, n_grid)
        maps.append(logdet.astype(np.float32))
    matrix = np.nanmean(np.stack(maps, axis=0), axis=0)

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(finite))
    if not np.isfinite(vmax):
        vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    im = ax.imshow(
        matrix,
        aspect="equal",
        origin="lower",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log det(I_z)")
    tick_idx = np.linspace(0, n_grid - 1, 5, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{axis[idx]:.1f}" for idx in tick_idx])
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{axis[idx]:.1f}" for idx in tick_idx])
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"log det(I_z) Colormap (mean over {len(seed_refs)} seed(s))")
    fig.tight_layout()
    _save_figure(fig, figures_dir / "I_z_t_colormap", figure_formats)
    plt.close(fig)


def _plot_trajectory_coverage(
    figures_dir: Path,
    *,
    records: list[dict[str, Any]],
    figure_formats: Sequence[str],
) -> None:
    ref = _reference_record(records)
    if ref is None:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    metadata = dict(ref["metadata"])
    system_id_raw = metadata.get("system_id")
    system_id = (
        str(system_id_raw).strip()
        if isinstance(system_id_raw, str) and str(system_id_raw).strip()
        else ("bistable_attractor" if metadata.get("hard_setup") else "single_attractor")
    )
    if not has_planar_system_spec(system_id):
        return
    system_spec = get_planar_system_spec(system_id)
    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=np.float32)
    dynamics_alpha = float(metadata.get("dynamics_alpha", 0.7))
    env_preset = get_environment_preset_from_metadata(metadata)
    grid_lim = float(env_preset.x_range)
    dyn_true = PlanarResidualDynamics(
        system_id=system_spec.system_id,
        embedding=theta_true,
        residual_fn=residual_torch,
        dynamics_alpha=dynamics_alpha,
        device="cpu",
    )

    policy_ids = sorted({str(record["policy_id"]) for record in records})
    grouped: dict[str, list[np.ndarray]] = {policy_id: [] for policy_id in policy_ids}
    for record in records:
        trace_path = _trace_path(
            record,
            metadata_key="state_action_trace_path",
            fallback_name="state_action_trace.csv",
        )
        if trace_path is None:
            continue
        pts: list[tuple[float, float]] = []
        for row in _read_trace_csv(trace_path):
            x_val = safe_float(row.get("true_x"))
            v_val = safe_float(row.get("true_v"))
            if x_val is None or v_val is None:
                continue
            pts.append((x_val, v_val))
        if pts:
            grouped[str(record["policy_id"])].append(np.asarray(pts, dtype=np.float32))

    n_panels = len(policy_ids)
    if n_panels == 0:
        return
    n_cols = 2 if n_panels > 1 else 1
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.2 * n_cols, 6.4 * n_rows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, max(n_panels, 1)))

    for idx, policy_id in enumerate(policy_ids):
        ax = axes[idx // n_cols, idx % n_cols]
        color = colors[idx]
        plot_vector_field(
            dyn_true,
            ax=ax,
            x_range=grid_lim,
            n_grid=28,
            is_residual=True,
            device="cpu",
        )
        trajs = grouped.get(policy_id, [])
        for traj in trajs:
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2, alpha=0.8)
        if trajs:
            starts = np.asarray([traj[0] for traj in trajs if traj.shape[0] > 0], dtype=np.float32)
            ax.scatter(
                starts[:, 0],
                starts[:, 1],
                s=16,
                color=color,
                alpha=1.0,
                edgecolors="none",
                zorder=5,
            )
        decorate_phase_space_axis(
            ax,
            xlim=(-grid_lim, grid_lim),
            ylim=(-grid_lim, grid_lim),
            title=f"{policy_id} (n={len(trajs)})",
            grid_alpha=0.20,
        )

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    fig.suptitle(f"Trajectory Coverage on True {system_spec.label} Vector Field", y=0.98)
    fig.tight_layout()
    _save_figure(fig, figures_dir / "trajectory_coverage_by_policy", figure_formats)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize COSYNE v2 experiments")
    parser.add_argument("--base-dir", type=str, default="results/cosyne")
    parser.add_argument("--exp-id", choices=list_experiment_ids(), required=True)
    parser.add_argument("--summary-dir", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="0,10,20,30")
    parser.add_argument(
        "--figure-formats",
        type=str,
        default=".pdf",
        help="Comma-separated figure extensions to save, e.g. '.pdf' or 'pdf,svg'.",
    )
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        figure_formats = _parse_figure_formats(args.figure_formats)
    except ValueError as exc:
        parser.error(str(exc))
    exp_spec = get_experiment_spec(str(args.exp_id))
    base_dir = resolve_session_root(Path(args.base_dir), create=False, exp_ids=[exp_spec.exp_id])
    summary_dir = (
        Path(args.summary_dir) if args.summary_dir else base_dir / exp_spec.exp_id / "summary"
    )
    seeds = parse_csv_ints(args.seeds) or [0, 10, 20, 30]
    records, missing = collect_track_records(base_dir, exp_spec.exp_id, seeds)
    value_key = (
        "embedding_error_final"
        if exp_spec.summary_value_kind == "parameter_error"
        else "dynamics_mse_final"
    )
    trace_key = (
        "parameter_error_trace_path"
        if exp_spec.summary_value_kind == "parameter_error"
        else "dynamics_mse_trace_path"
    )
    trace_name = (
        "parameter_error_trace.csv"
        if exp_spec.summary_value_kind == "parameter_error"
        else "dynamics_mse_trace.csv"
    )
    trace_col = (
        "parameter_error" if exp_spec.summary_value_kind == "parameter_error" else "dynamics_mse"
    )
    value_prefix = (
        "parameter_error" if exp_spec.summary_value_kind == "parameter_error" else "dynamics_mse"
    )
    rows = collect_track_rows(records, value_key=value_key)
    trace_rows = aggregate_trace(
        records, metadata_key=trace_key, fallback_name=trace_name, value_col=trace_col
    )
    traj_rows = aggregate_trace(
        records,
        metadata_key="trajectory_r2_trace_path",
        fallback_name="trajectory_r2_trace.csv",
        value_col="trajectory_r2",
    )
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
    _write_csv(
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
    _write_curve_csv(
        summary_dir / "parameter_covariance_trace_over_steps.csv",
        cov_rows,
        "parameter_covariance_trace_mean",
    )
    _write_curve_csv(summary_dir / "I_z_t_over_steps.csv", info_rows, "I_z_t_mean")
    _write_markdown(
        summary_dir / "metrics.md", exp_spec.exp_id, rows, missing, exp_spec.summary_value_label
    )
    figures_dir = summary_dir / "figures"
    _plot_curves(
        figures_dir,
        rows=rows,
        trace_rows=trace_rows,
        cov_rows=cov_rows,
        traj_rows=traj_rows,
        value_label=exp_spec.summary_value_label,
        value_prefix=value_prefix,
        figure_formats=figure_formats,
    )
    for plotter in (
        _plot_neuron_tuning_curve_colormap,
        _plot_information_colormap,
        _plot_trajectory_coverage,
    ):
        try:
            plotter(
                figures_dir,
                records=records,
                figure_formats=figure_formats,
            )
        except Exception:
            pass
    print(f"Wrote {len(rows)} rows to {summary_dir / 'metrics.csv'}")
    if missing and args.fail_on_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
