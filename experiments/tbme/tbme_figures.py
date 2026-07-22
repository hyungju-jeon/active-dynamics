#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..experiment_definitions import get_environment_preset
from ..experiment_io import experiment_env_slug
from .figures import artifacts as _artifacts
from .figures import trajectories as _trajectories
from .figures import data as _data
from .figures import theme as _theme
from .run_tbme_experiments import configure_tbme_catalogs, shared_tbme_group_suites

configure_tbme_catalogs()


# Shared configuration (implementations live in experiments/tbme/figures/)
_TBME_STROKE_COLOR = _theme.STROKE_COLOR
_TBME_GRID_COLOR = _theme.GRID_COLOR

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_ROOT = _REPO_ROOT / "results"
_TBME_RESULTS_DIR = _RESULTS_ROOT / "tbme"


# GROUPS is evaluated at import time, so this constructor stays before GROUPS.
def _latest_session(base: Path) -> Path:
    sessions = [
        path
        for path in base.glob("session_*")
        if path.is_dir() and path.name.removeprefix("session_").isdigit()
    ]
    if not sessions:
        return base / "session_1"
    return max(sessions, key=lambda path: int(path.name.removeprefix("session_")))


@dataclass(frozen=True)
class SuiteRef:
    suite_id: str
    label: str
    session_root: Path
    slug: str
    policy_ids: tuple[str, ...] = ()


def _tbme_label(env_preset_id: str) -> str:
    env_preset = get_environment_preset(env_preset_id)
    return str(getattr(env_preset, "system_label", None) or env_preset.system_id)


def _build_groups(results_dir: Path) -> dict[str, list[SuiteRef]]:
    session_root = _latest_session(results_dir)
    groups: dict[str, list[SuiteRef]] = {}
    for group_name, entries in shared_tbme_group_suites().items():
        refs: list[SuiteRef] = []
        for entry in entries:
            env_preset_id = str(entry["env_preset_id"])
            refs.append(
                SuiteRef(
                    str(entry["suite_id"]),
                    _tbme_label(env_preset_id),
                    session_root,
                    experiment_env_slug(env_preset_id),
                    tuple(str(policy_id) for policy_id in entry["policy_ids"]),
                )
            )
        groups[group_name] = refs
    return groups


GROUPS: dict[str, list[SuiteRef]] = _build_groups(_TBME_RESULTS_DIR)


def _set_tbme_results_dir(results_dir: Path) -> None:
    global _TBME_RESULTS_DIR, GROUPS
    _TBME_RESULTS_DIR = Path(results_dir).resolve()
    GROUPS = _build_groups(_TBME_RESULTS_DIR)


POLICY_LABELS = _theme.POLICY_LABELS
POLICY_ORDER = _theme.POLICY_ORDER
POLICY_COLORS = _theme.POLICY_COLORS
FALLBACK_COLORS = _theme.FALLBACK_COLORS


# Shared TBME helpers (aliases; implementations live in experiments/tbme/figures/)

_apply_style = _theme.apply_style
_style_manuscript_axis = _theme.style_axis
_policy_sort_key = _theme.policy_sort_key
_policy_label = _theme.policy_label
_policy_color = _theme.policy_color
_r2_threshold_suffix = _data.r2_threshold_suffix
_write_csv = _artifacts.write_csv
_write_text = _artifacts.write_text
_unique_paths = _artifacts.unique_paths


def _suite_dir(group_name: str, suite_id: str) -> Path:
    for ref in GROUPS[group_name]:
        if ref.suite_id == suite_id:
            return ref.session_root / "tracks" / ref.suite_id
    raise KeyError(f"Unknown suite {group_name}/{suite_id}")


_tbme_trajectory_layout = _trajectories.trajectory_layout
_tbme_trajectory_plot_limit = _trajectories.trajectory_plot_limit
_tbme_trajectory_seed_color_map = _trajectories.trajectory_seed_color_map
_tbme_format_trajectory_axis = _trajectories.format_trajectory_axis
_tbme_trajectory_histogram = _trajectories.trajectory_histogram
_tbme_trajectory_density_cmap = _trajectories.trajectory_density_cmap
plot_trajectory_overlay = _trajectories.plot_trajectory_overlay
plot_trajectory_density = _trajectories.plot_trajectory_density


def summary_main(argv: list[str] | None = None) -> int:
    """Run TBME summary figure generation via the split summary module."""
    from .tbme_figures_summary import summary_main as _summary_main

    return _summary_main(argv)


def experiment_main(argv: list[str] | None = None) -> int:
    """Run TBME experiment-level figure generation via the split experiment module."""
    from .tbme_figures_experiment import experiment_main as _experiment_main

    return _experiment_main(argv)


def assets_main(argv: list[str] | None = None) -> int:
    """Run TBME manuscript asset assembly via the split asset module."""
    from .tbme_figures_assets import assets_main as _assets_main

    return _assets_main(argv)


def diagnostics_main(argv: list[str] | None = None) -> int:
    """Run TBME environment diagnostics via the split diagnostics module."""
    from .tbme_figures_diagnostics import main as _diagnostics_main

    return _diagnostics_main(argv)
