"""Experiment-level figure CLI: dispatches the suite-based figure families."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from . import ablation as _ablation
from . import bottleneck as _bottleneck
from . import mismatch as _mismatch
from . import recovery as _recovery
from . import true_dynamics as _true_dynamics
from .artifacts import unique_paths as _unique_paths
from .groups import groups as _groups_table, suite_dir as _suite_dir


_PLOTS = (
    "bottleneck_sweep",
    "objective_ablation",
    "mismatch_dose_response",
    "true_dynamics_all",
    "per_parameter_recovery",
)
EXPERIMENT_PLOTS = _PLOTS

_OBJECTIVE_DEFINITION_PLOTS = {"objective_ablation"}
_REQUIRED_SUITES_BY_PLOT = {
    "bottleneck_sweep": _bottleneck.REQUIRED_SUITES,
    "objective_ablation": _ablation.REQUIRED_SUITES,
    "mismatch_dose_response": _mismatch.required_suites(),
}


def _parse_plots(raw: str) -> list[str]:
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


def _required_suite_dirs(plot_ids: Sequence[str]) -> list[Path]:
    suite_keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for plot_id in plot_ids:
        for key in _REQUIRED_SUITES_BY_PLOT.get(plot_id, ()):
            if key not in seen:
                suite_keys.append(key)
                seen.add(key)
    return [_suite_dir(group_name, suite_id) for group_name, suite_id in suite_keys]


def _build_parser() -> argparse.ArgumentParser:
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
        default=",".join(_groups_table()),
        help="Comma-separated TBME groups whose suite folders receive global experiment figures.",
    )
    return parser


def _suite_dirs_from_groups(raw: str) -> list[Path]:
    group_ids = [item.strip() for item in str(raw).split(",") if item.strip()]
    unknown = sorted(set(group_ids) - set(_groups_table()))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")
    if not group_ids:
        raise ValueError("At least one TBME group is required")
    return _unique_paths(
        ref.session_root / "tracks" / ref.suite_id
        for group_id in group_ids
        for ref in _groups_table()[group_id]
    )


def experiment_main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    max_seeds = int(args.max_seeds)
    plot_ids = _parse_plots(str(args.plots))
    output_suite_dirs = _suite_dirs_from_groups(str(args.groups))
    missing_suite_dirs = sorted(
        (
            suite_dir
            for suite_dir in _required_suite_dirs(plot_ids)
            if not suite_dir.exists()
        ),
        key=str,
    )
    if missing_suite_dirs:
        missing_text = ", ".join(str(path) for path in missing_suite_dirs)
        raise FileNotFoundError(f"Missing TBME experiment suite(s): {missing_text}")

    written: list[Path] = []
    if any(plot_id in _OBJECTIVE_DEFINITION_PLOTS for plot_id in plot_ids):
        written.extend(
            _ablation.write_definition_tables(
                _required_suite_dirs(["objective_ablation"])
            )
        )

    experiment_plotters = {
        "bottleneck_sweep": _bottleneck.generate,
        "objective_ablation": _ablation.generate,
        "mismatch_dose_response": _mismatch.generate,
    }
    figure_only_plotters = {
        "true_dynamics_all": lambda: _true_dynamics.generate(output_suite_dirs),
        "per_parameter_recovery": lambda: _recovery.generate(
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
