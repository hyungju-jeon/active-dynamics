#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .figures import ablation as _ablation
from .figures import bottleneck as _bottleneck
from .figures import gates as _gates
from .figures import mismatch as _mismatch
from .figures import information as _information
from .figures import records as _records
from .figures import recovery as _recovery
from .figures import true_dynamics as _true_dynamics
from .figures import artifacts as _fig_artifacts
from .figures import data as _fig_data
from .figures import theme as _fig_theme
from .figures.groups import SuiteSource
from .figures.records import RunRecord
from .tbme_figures import (
    GROUPS,
    _suite_dir,
    _unique_paths,
)

# Experiment manuscript output
_experiment_C_STROKE = _fig_theme.STROKE_COLOR
_experiment_C_NEUTRAL_LIGHT = _fig_theme.NEUTRAL_LIGHT
_experiment_C_NEUTRAL_FILL = _fig_theme.NEUTRAL_FILL
_experiment_C_GRID = _fig_theme.GRID_COLOR

_experiment_BOTTLENECK_POLICIES = _bottleneck.BOTTLENECK_POLICIES
_experiment_OBJECTIVE_POLICIES = _ablation.OBJECTIVE_POLICIES
_experiment_OBJECTIVE_DEFINITIONS = _ablation.OBJECTIVE_DEFINITIONS
_experiment_DOSE_POLICIES = _mismatch.DOSE_POLICIES


_ExperimentSuiteSource = SuiteSource


_ExperimentRunRecord = RunRecord


_experiment_PLOTS = (
    "bottleneck_sweep",
    "objective_ablation",
    "mismatch_dose_response",
    "true_dynamics_all",
    "per_parameter_recovery",
)
EXPERIMENT_PLOTS = _experiment_PLOTS

_experiment_OBJECTIVE_DEFINITION_PLOTS = {"objective_ablation"}
_experiment_REQUIRED_SUITES_BY_PLOT = {
    "bottleneck_sweep": _bottleneck.REQUIRED_SUITES,
    "objective_ablation": _ablation.REQUIRED_SUITES,
    "mismatch_dose_response": _mismatch.REQUIRED_SUITES,
}


_experiment_artifact_paths = _fig_artifacts.artifact_paths
_experiment_write_csv_artifacts = _fig_artifacts.write_csv_artifacts
_experiment_write_text_artifacts = _fig_artifacts.write_text_artifacts
_experiment_copy_artifact = _fig_artifacts.copy_artifact


_style_experiment_axis = _fig_theme.style_experiment_axis


plot_bottleneck_sweep = _bottleneck.plot_bottleneck_sweep


plot_objective_ablation = _ablation.plot_objective_ablation


plot_mismatch_dose_response = _mismatch.plot_mismatch_dose_response


plot_neutral_vector_field = _gates.plot_neutral_vector_field


plot_per_parameter_recovery = _recovery.plot_per_parameter_recovery


def _experiment_parse_plots(raw: str) -> list[str]:
    plot_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(plot_ids) - set(EXPERIMENT_PLOTS))
    if unknown:
        raise ValueError(f"Unknown experiment plot(s): {', '.join(unknown)}")

    ordered: list[str] = []
    seen: set[str] = set()
    for plot_id in plot_ids:
        if plot_id not in seen:
            ordered.append(plot_id)
            seen.add(plot_id)
    return ordered


def _experiment_required_suite_dirs(plot_ids: Sequence[str]) -> list[Path]:
    suite_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for plot_id in plot_ids:
        for key in _experiment_REQUIRED_SUITES_BY_PLOT.get(plot_id, ()):
            if key not in seen:
                suite_keys.append(key)
                seen.add(key)
    return [_suite_dir(group_name, suite_id) for group_name, suite_id in suite_keys]


_experiment_policy_label = _fig_theme.short_policy_label


_experiment_metrics_by_policy = _fig_data.metrics_by_policy
_experiment_metric_values = _fig_data.metric_values
_experiment_metric_mean_sem = _fig_data.metric_mean_sem
_experiment_curve_rows = _fig_data.curve_rows
_experiment_r2_threshold_step = _fig_data.r2_threshold_step
_experiment_r2_threshold_times = _fig_data.r2_threshold_times


_experiment_plot_bottleneck_sweep = _bottleneck.generate


_experiment_objective_sources = _ablation.objective_sources
_experiment_write_objective_definition_tables = _ablation.write_definition_tables
_experiment_plot_objective_ablation = _ablation.generate


_experiment_dose_sources = _mismatch.dose_sources
_experiment_plot_mismatch_dose_response = _mismatch.generate


_experiment_collect_records = _records.collect_records


def _experiment_build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME experiment-level figures into suite result folders.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=100,
        help="Maximum seeds per policy to read for trace-derived figures.",
    )
    parser.add_argument(
        "--plots",
        type=str,
        default=",".join(EXPERIMENT_PLOTS),
        help="Comma-separated TBME experiment plot ids.",
    )
    parser.add_argument(
        "--groups",
        type=str,
        default=",".join(GROUPS),
        help="Comma-separated TBME groups whose suite folders receive global experiment figures.",
    )
    return parser


def _experiment_suite_dirs_from_groups(raw: str) -> list[Path]:
    group_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(group_ids) - set(GROUPS))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    if not group_ids:
        raise ValueError("At least one TBME group is required")
    return _unique_paths(
        ref.session_root / "tracks" / ref.suite_id
        for group_id in group_ids
        for ref in GROUPS[group_id]
    )


_experiment_short_policy_label = _fig_theme.extended_policy_label


_experiment_state_bounds_from_metadata = _records.state_bounds_from_metadata
_experiment_trace_path = _records.record_trace_path
_experiment_load_xy_trace = _records.load_xy_trace
_experiment_logdet_information = _information.logdet_information
_experiment_observation_model_key = _information.observation_model_key
_experiment_information_reference_records = _information.information_reference_records
_experiment_make_information_grid = _information.make_information_grid
_experiment_make_mean_information_grid = _information.make_mean_information_grid


_experiment_plot_true_dynamics_all = _true_dynamics.generate


_experiment_aggregate_parameter_traces = _recovery.aggregate_parameter_traces
_experiment_plot_per_parameter_recovery = _recovery.generate


_COMPOUND_POLICY_ORDER = _gates.COMPOUND_POLICY_ORDER
_COMPOUND_POLICY_LABELS = _gates.COMPOUND_POLICY_LABELS
_COMPOUND_POLICY_COLORS = _gates.COMPOUND_POLICY_COLORS
_compound_trace_records = _gates.compound_trace_records
_compound_curve = _gates.compound_curve
_compound_summary_rows = _gates.compound_summary_rows
_selector_gate_occupancy = _gates.selector_gate_occupancy
_compound_paired_summary_rows = _gates.compound_paired_summary_rows
_reach_hold_selector_occupancy = _gates.reach_hold_selector_occupancy
generate_compound_tri_gate_figures = _gates.generate_compound_tri_gate_figures
_compound_poisson_observation_figure = _gates._compound_poisson_observation_figure
generate_three_gate_diagnostic_figures = _gates.generate_three_gate_diagnostic_figures


def experiment_main(argv: list[str] | None = None) -> int:
    args = _experiment_build_parser().parse_args(argv)
    max_seeds = int(args.max_seeds)
    plot_ids = _experiment_parse_plots(str(args.plots))
    output_suite_dirs = _experiment_suite_dirs_from_groups(str(args.groups))
    missing_suite_dirs = sorted(
        (
            suite_dir
            for suite_dir in _experiment_required_suite_dirs(plot_ids)
            if not suite_dir.exists()
        ),
        key=str,
    )
    if missing_suite_dirs:
        missing_text = ", ".join(str(path) for path in missing_suite_dirs)
        raise FileNotFoundError(f"Missing TBME experiment suite(s): {missing_text}")

    written: list[Path] = []
    if any(plot_id in _experiment_OBJECTIVE_DEFINITION_PLOTS for plot_id in plot_ids):
        written.extend(
            _experiment_write_objective_definition_tables(
                _experiment_required_suite_dirs(["objective_ablation"])
            )
        )

    experiment_plotters = {
        "bottleneck_sweep": _experiment_plot_bottleneck_sweep,
        "objective_ablation": _experiment_plot_objective_ablation,
        "mismatch_dose_response": _experiment_plot_mismatch_dose_response,
    }
    figure_only_plotters = {
        "true_dynamics_all": lambda: _experiment_plot_true_dynamics_all(output_suite_dirs),
        "per_parameter_recovery": lambda: _experiment_plot_per_parameter_recovery(
            max_seeds=max_seeds
        ),
    }
    for plot_id in plot_ids:
        if plot_id in experiment_plotters:
            written.extend(experiment_plotters[plot_id]())
        else:
            written.extend(figure_only_plotters[plot_id]())

    for path in written:
        print(path)
    return 0
