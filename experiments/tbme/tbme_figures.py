#!/usr/bin/env python3
"""Backward-compatibility shim — implementations live in experiments/tbme/figures/."""
from __future__ import annotations

from pathlib import Path

from .figures import artifacts as _artifacts
from .figures import data as _data
from .figures import groups as _groups
from .figures import theme as _theme
from .figures import trajectories as _trajectories
from .figures.groups import SuiteRef  # noqa: F401
from .run_tbme_experiments import configure_tbme_catalogs

configure_tbme_catalogs()

_REPO_ROOT = _groups.REPO_ROOT
_RESULTS_ROOT = _groups.RESULTS_ROOT
_TBME_STROKE_COLOR = _theme.STROKE_COLOR
_TBME_GRID_COLOR = _theme.GRID_COLOR

_latest_session = _groups.latest_session
_build_groups = _groups._build_groups
_set_tbme_results_dir = _groups.set_results_dir
_suite_dir = _groups.suite_dir

POLICY_LABELS = _theme.POLICY_LABELS
POLICY_ORDER = _theme.POLICY_ORDER
POLICY_COLORS = _theme.POLICY_COLORS
FALLBACK_COLORS = _theme.FALLBACK_COLORS

_apply_style = _theme.apply_style
_style_manuscript_axis = _theme.style_axis
_policy_sort_key = _theme.policy_sort_key
_policy_label = _theme.policy_label
_policy_color = _theme.policy_color
_r2_threshold_suffix = _data.r2_threshold_suffix
_write_csv = _artifacts.write_csv
_write_text = _artifacts.write_text
_unique_paths = _artifacts.unique_paths

_tbme_trajectory_layout = _trajectories.trajectory_layout
_tbme_trajectory_plot_limit = _trajectories.trajectory_plot_limit
_tbme_trajectory_seed_color_map = _trajectories.trajectory_seed_color_map
_tbme_format_trajectory_axis = _trajectories.format_trajectory_axis
_tbme_trajectory_histogram = _trajectories.trajectory_histogram
_tbme_trajectory_density_cmap = _trajectories.trajectory_density_cmap
plot_trajectory_overlay = _trajectories.plot_trajectory_overlay
plot_trajectory_density = _trajectories.plot_trajectory_density


def __getattr__(name: str):
    # GROUPS and the results dir are mutable module state in figures.groups;
    # resolve them dynamically so --results-dir overrides stay visible here.
    if name == "GROUPS":
        return _groups.groups()
    if name == "_TBME_RESULTS_DIR":
        return _groups.results_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def summary_main(argv: list[str] | None = None) -> int:
    """Run TBME summary figure generation via the figures package."""
    from .figures.summary import summary_main as _summary_main

    return _summary_main(argv)


def experiment_main(argv: list[str] | None = None) -> int:
    """Run TBME experiment-level figure generation via the figures package."""
    from .tbme_figures_experiment import experiment_main as _experiment_main

    return _experiment_main(argv)


def assets_main(argv: list[str] | None = None) -> int:
    """Run TBME manuscript asset assembly via the split asset module."""
    from .tbme_figures_assets import assets_main as _assets_main

    return _assets_main(argv)


def diagnostics_main(argv: list[str] | None = None) -> int:
    """Run TBME environment diagnostics via the figures package."""
    from .figures.diagnostics import main as _diagnostics_main

    return _diagnostics_main(argv)
