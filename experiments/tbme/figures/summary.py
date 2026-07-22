"""Per-suite summary figure family.

Metric-vs-time and trajectory overlay/density figures for each result suite.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from actdyn.environment.vectorfield import ResidualDynamicsCallable
from actdyn.utils.experiment_runtime import read_trace_csv
from actdyn.utils.figure_io import parse_figure_formats, sample_sem

from ...experiment_io import (
    experiment_env_slug,
    get_environment_preset_from_metadata,
    load_json,
)
from ...visualize import (
    plot_final_value_by_policy,
    plot_metric_over_cpu_time,
    plot_metric_over_steps,
)
from .. import tbme_figures as _figures
from ..tbme_figures import (
    _TBME_STROKE_COLOR,
    _apply_style,
    _policy_color,
    _policy_label,
    _policy_sort_key,
    _style_manuscript_axis,
    plot_trajectory_density,
    plot_trajectory_overlay,
)
from ..tbme_io import read_state_action_trace, read_xy_trace as _read_xy_trace

# Summary output
_summary_trace_C_WRITE = "#1F4FA8"
_summary_trace_C_STROKE = "#3A3A3A"
_summary_trace_C_NEUTRAL_LIGHT = "#C8C1B8"

SUMMARY_POLICY_FAMILIES = {
    "adaptive": (
        "active_planning",
        "adaptive",
        "adaptive_async_anytime",
        "adaptive_async_realtime",
    ),  # Total 4 policies
    "baselines": (
        "active_planning",
        "active_myopic",
        "prbs",
        "random",
        "flex",
        "flex_true_state",
        "rhc",
        "off_policy",
    ),  # Total 8 policies
    "objective": (
        "active_myopic",
        "active_planning",
        "active_fully_observable",
        "active_e_optimality",
        "active_state_information",
        "active_dynamics",
        "active_observation_variance",
        "active_state_variance",
    ),  # Total 8 policies
    "scheduling": (
        "adaptive",
        "adaptive_async_anytime",
        "active_planning",
        "active_planning_u1_r1_h40",
        "active_planning_u5_r5_h40",
        "active_planning_u5_r10_h40",
        "active_planning_u10_r10_h40",
        "active_planning_u5_r20_h40",
        "active_planning_u10_r20_h40",
    ),  # Total 9 policies
}


def _summary_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write current TBME summary and trajectory figures."
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="",
        help="Comma-separated env_preset_id:policy_id pairs.",
    )
    parser.add_argument("--figure-formats", type=str, default=".pdf")
    parser.add_argument("--trajectory-max-seeds", type=int, default=50)
    parser.add_argument("--density-bins", type=int, default=96)
    return parser


def _parse_selection(
    raw: str,
) -> tuple[list[Path], dict[Path, set[str]]]:
    session_root = _figures._latest_session(_figures._TBME_RESULTS_DIR)
    suite_dirs: list[Path] = []
    policy_ids_by_suite: dict[Path, set[str]] = {}
    for item in (part.strip() for part in str(raw).split(",") if part.strip()):
        parts = item.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid summary selection item: {item!r}")
        env_preset_id, policy_id = parts
        if not env_preset_id or not policy_id:
            raise ValueError(f"Invalid summary selection item: {item!r}")
        suite_dir = session_root / "tracks" / experiment_env_slug(env_preset_id)
        if suite_dir not in policy_ids_by_suite:
            suite_dirs.append(suite_dir)
            policy_ids_by_suite[suite_dir] = set()
        policy_ids_by_suite[suite_dir].add(policy_id)
    return suite_dirs, policy_ids_by_suite


def _get_policy_families(policy_ids: set[str]) -> list[tuple[str, list[str]]]:
    families: list[tuple[str, list[str]]] = []
    for family_id, family_policy_ids in SUMMARY_POLICY_FAMILIES.items():
        if set(family_policy_ids).issubset(policy_ids):
            families.append((family_id, list(family_policy_ids)))
    remaining = sorted(
        policy_ids.difference(
            policy_id
            for family_policy_ids in SUMMARY_POLICY_FAMILIES.values()
            for policy_id in family_policy_ids
        ),
        key=_policy_sort_key,
    )
    if remaining:
        families.append(("other", remaining))
    return families


def _load_state_action(
    suite_dir: Path,
    *,
    max_seeds: int | None = None,
    policy_ids: Sequence[str] | None = None,
) -> list[tuple[str, int, Path]]:
    track_dir = suite_dir
    if not track_dir.exists():
        return []
    paths: list[tuple[str, int, Path]] = []
    policy_filter = set(policy_ids) if policy_ids is not None else None
    for policy_dir in sorted(
        (p for p in track_dir.iterdir() if p.is_dir()),
        key=lambda p: _policy_sort_key(p.name),
    ):
        if policy_filter is not None and policy_dir.name not in policy_filter:
            continue
        seed_dirs: list[tuple[int, Path]] = []
        for seed_dir in policy_dir.glob("seed_*"):
            suffix = seed_dir.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), seed_dir))
        for seed, seed_dir in sorted(seed_dirs)[:max_seeds]:
            trace_path = None
            for repeat_dir in sorted(seed_dir.glob("repeat_*")):
                candidate = repeat_dir / "state_action_trace.csv"
                if candidate.exists():
                    trace_path = candidate
                    break
            if trace_path is None:
                continue
            paths.append((policy_dir.name, seed, trace_path))
    return paths


def _group_trajectories(
    suite_dir: Path,
    *,
    max_seeds: int,
    policy_ids: Sequence[str] | None = None,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
    for policy_id, seed, trace_path in _load_state_action(
        suite_dir,
        max_seeds=max_seeds,
        policy_ids=policy_ids,
    ):
        traj = _read_xy_trace(trace_path)
        if traj.size:
            grouped.setdefault(policy_id, []).append((seed, traj))
    return grouped


def _get_action_magnitude(
    suite_dir: Path,
    *,
    policy_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    samples: dict[tuple[str, int], list[float]] = {}
    for policy_id, _seed, trace_path in _load_state_action(
        suite_dir,
        policy_ids=policy_ids,
    ):
        steps, _true_state, _model_state, actions = read_state_action_trace(trace_path)
        magnitudes = np.linalg.norm(actions, axis=1)
        for step, magnitude in zip(steps, magnitudes, strict=True):
            if np.isfinite(magnitude):
                samples.setdefault((policy_id, int(step)), []).append(float(magnitude))

    rows: list[dict[str, Any]] = []
    for policy_id, step in sorted(samples, key=lambda key: (_policy_sort_key(key[0]), key[1])):
        values = samples[(policy_id, step)]
        rows.append(
            {
                "policy_id": policy_id,
                "step": step,
                "value_mean": float(np.mean(values)),
                "value_sem": sample_sem(values),
            }
        )
    return rows


def _dynamics_from_metadata(metadata: dict[str, Any]) -> tuple[Any, float, str] | None:
    env_preset = get_environment_preset_from_metadata(metadata)
    if bool(getattr(env_preset, "real_data", False)):
        return None
    theta_true = np.asarray(metadata.get("embedding_true", [0.0, 0.0]), dtype=np.float32)
    dynamics_alpha = float(metadata.get("dynamics_alpha", 0.7))
    grid_lim = float(env_preset.resolved_plot_limit())
    dyn_true = ResidualDynamicsCallable(
        dynamics_type=env_preset.resolved_dynamics_type(),
        dyn_params=env_preset.params_from_embedding(theta_true),
        dynamics_alpha=dynamics_alpha,
        device="cpu",
    )
    label = str(getattr(env_preset, "system_label", None) or env_preset.system_id)
    return dyn_true, grid_lim, label


def make_summary_metric_figures(
    suite_dir: Path,
    figure_formats: Sequence[str],
    *,
    policy_ids: Sequence[str] | None = None,
    family_id: str,
) -> list[Path]:
    summary_dir = suite_dir / "summary"
    figures_dir = summary_dir / "figures"
    policy_filter = set(policy_ids) if policy_ids is not None else None
    family_label = family_id.replace("_", " ").title()

    value_list = [
        "parameter_error",
        "trajectory_r2",
        # "parameter_covariance_trace",
    ]
    rows = [
        row
        for row in read_trace_csv(summary_dir / "metrics.csv")
        if policy_filter is None or str(row.get("policy_id")) in policy_filter
    ]

    for value_prefix in value_list:
        value_label = value_prefix.replace("_", " ").title()
        value_column = f"{value_prefix}_mean"
        final_column = (
            "trajectory_r2_final_mean" if value_prefix == "trajectory_r2" else "value_final_mean"
        )
        final_rows = [
            {**row, "value_final_mean": row[final_column]}
            for row in rows
            if final_column in row
        ]
        trace_rows = [
            {**row, "value_mean": row[value_column]}
            for row in read_trace_csv(summary_dir / f"{value_prefix}_over_steps.csv")
            if value_column in row
            and (policy_filter is None or str(row.get("policy_id")) in policy_filter)
        ]
        plot_final_value_by_policy(
            figures_dir,
            rows=final_rows,
            ylabel=f"Final {value_label} (mean +/- SEM over seeds)",
            title=f"{value_label} by Policy ({family_label})",
            output_stem=f"final_{value_prefix}_by_policy_{family_id}",
            figure_formats=figure_formats,
            policy_sort_key=_policy_sort_key,
            policy_label=_policy_label,
            policy_color=_policy_color,
            apply_style=_apply_style,
            style_axis=_style_manuscript_axis,
            stroke_color=_TBME_STROKE_COLOR,
        )
        plot_metric_over_steps(
            figures_dir,
            rows=trace_rows,
            ylabel=f"{value_label} (mean ± SEM)",
            title=f"{value_label} Over Steps ({family_label})",
            output_stem=f"{value_prefix}_over_steps_{family_id}",
            figure_formats=figure_formats,
            policy_sort_key=_policy_sort_key,
            policy_label=_policy_label,
            policy_color=_policy_color,
            apply_style=_apply_style,
            style_axis=_style_manuscript_axis,
        )
        plot_metric_over_cpu_time(
            figures_dir,
            rows=trace_rows,
            ylabel=f"{value_label} (mean ± SEM)",
            title=f"{value_label} Over Loop Compute Time ({family_label})",
            output_stem=f"{value_prefix}_over_cpu_time_{family_id}",
            figure_formats=figure_formats,
            policy_sort_key=_policy_sort_key,
            policy_label=_policy_label,
            policy_color=_policy_color,
            apply_style=_apply_style,
            style_axis=_style_manuscript_axis,
        )

    action_rows = _get_action_magnitude(suite_dir, policy_ids=policy_ids)
    plot_metric_over_steps(
        figures_dir,
        rows=action_rows,
        ylabel="Action Magnitude (mean ± SEM)",
        title=f"Action Magnitude Over Steps ({family_label})",
        output_stem=f"action_magnitude_over_steps_{family_id}",
        figure_formats=figure_formats,
        policy_sort_key=_policy_sort_key,
        policy_label=_policy_label,
        policy_color=_policy_color,
        apply_style=_apply_style,
        style_axis=_style_manuscript_axis,
    )

    stems = [
        stem
        for value_prefix in value_list
        for stem in (
            f"final_{value_prefix}_by_policy",
            f"{value_prefix}_over_steps",
            f"{value_prefix}_over_cpu_time",
        )
    ]
    stems.append("action_magnitude_over_steps")

    return [
        figures_dir / f"{stem}_{family_id}{fmt}"
        for stem in stems
        for fmt in figure_formats
        if (figures_dir / f"{stem}_{family_id}{fmt}").exists()
    ]


def make_summary_trajectory_figures(
    suite_dir: Path,
    figure_formats: Sequence[str],
    *,
    max_seeds: int,
    density_bins: int,
    policy_ids: Sequence[str] | None = None,
    family_id: str,
) -> list[Path]:
    metadata_paths = sorted(suite_dir.glob("*/seed_*/repeat_*/run_metadata.json"))
    if not metadata_paths:
        return []

    metadata = load_json(metadata_paths[0])
    dynamics_payload = _dynamics_from_metadata(metadata)
    if dynamics_payload is None:
        return []
    dyn_true, grid_lim, system_label = dynamics_payload

    grouped = _group_trajectories(
        suite_dir,
        max_seeds=max_seeds,
        policy_ids=policy_ids,
    )
    if not grouped:
        return []
    figures_dir = suite_dir / "summary" / "figures"
    family_label = family_id.replace("_", " ").title()
    written: list[Path] = []
    written.extend(
        plot_trajectory_overlay(
            figures_dir,
            output_stem=f"trajectory_overlay_vectorfield_by_policy_{family_id}",
            figure_formats=figure_formats,
            suite_name=f"{suite_dir.name} ({family_label})",
            grouped=grouped,
            dyn_true=dyn_true,
            grid_lim=grid_lim,
            system_label=system_label,
            max_seeds=max_seeds,
            policy_sort_key=_policy_sort_key,
            policy_label=_policy_label,
            apply_style=_apply_style,
            stroke_color=_summary_trace_C_STROKE,
            write_color=_summary_trace_C_WRITE,
            neutral_light=_summary_trace_C_NEUTRAL_LIGHT,
        )
    )
    written.extend(
        plot_trajectory_density(
            figures_dir,
            output_stem=f"trajectory_density_by_policy_{family_id}",
            figure_formats=figure_formats,
            suite_name=f"{suite_dir.name} ({family_label})",
            grouped=grouped,
            dyn_true=dyn_true,
            grid_lim=grid_lim,
            system_label=system_label,
            max_seeds=max_seeds,
            bins=density_bins,
            policy_sort_key=_policy_sort_key,
            policy_label=_policy_label,
            apply_style=_apply_style,
            stroke_color=_summary_trace_C_STROKE,
            neutral_light=_summary_trace_C_NEUTRAL_LIGHT,
        )
    )
    return written


def summary_main(argv: list[str] | None = None) -> int:
    _apply_style()
    args = _summary_build_parser().parse_args(argv)
    figure_formats = parse_figure_formats(str(args.figure_formats))
    suite_dirs, policy_ids_by_suite = _parse_selection(str(args.selection))

    written: list[Path] = []
    for suite_dir in suite_dirs:
        if suite_dir.exists():
            for family_id, policy_ids in _get_policy_families(policy_ids_by_suite[suite_dir]):
                written.extend(
                    make_summary_metric_figures(
                        suite_dir,
                        figure_formats,
                        policy_ids=policy_ids,
                        family_id=family_id,
                    )
                )
                written.extend(
                    make_summary_trajectory_figures(
                        suite_dir,
                        figure_formats,
                        max_seeds=int(args.trajectory_max_seeds),
                        density_bins=int(args.density_bins),
                        policy_ids=policy_ids,
                        family_id=family_id,
                    )
                )
    for path in written:
        print(path)
    print(f"wrote {len(written)} summary figure files")
    return 0
