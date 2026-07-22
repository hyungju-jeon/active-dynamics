"""Suite/group resolution for TBME figures.

Owns the GROUPS table: which result suites exist per group, resolved against
the newest ``session_*`` directory under the results root. The table is built
lazily from the experiment catalog and rebuilt when ``set_results_dir`` points
at a different root (the ``--results-dir`` CLI override).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
DEFAULT_RESULTS_DIR = RESULTS_ROOT / "tbme"

_results_dir: Path = DEFAULT_RESULTS_DIR
_groups: dict[str, list["SuiteRef"]] | None = None


@dataclass(frozen=True)
class SuiteRef:
    suite_id: str
    label: str
    session_root: Path
    slug: str
    policy_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuiteSource:
    """One suite feeding a figure: an experiment id, display label, and data dir."""

    exp_id: str
    label: str
    suite_dir: Path
    dose: str | None = None
    family: str | None = None


def latest_session(base: Path) -> Path:
    sessions = [
        path
        for path in base.glob("session_*")
        if path.is_dir() and path.name.removeprefix("session_").isdigit()
    ]
    if not sessions:
        return base / "session_1"
    return max(sessions, key=lambda path: int(path.name.removeprefix("session_")))


def _suite_label(env_preset_id: str) -> str:
    from ...experiment_definitions import get_environment_preset

    env_preset = get_environment_preset(env_preset_id)
    return str(getattr(env_preset, "system_label", None) or env_preset.system_id)


def _build_groups(results_dir: Path) -> dict[str, list[SuiteRef]]:
    from ...experiment_io import experiment_env_slug
    from ..run_tbme_experiments import configure_tbme_catalogs, shared_tbme_group_suites

    configure_tbme_catalogs()
    session_root = latest_session(results_dir)
    groups_table: dict[str, list[SuiteRef]] = {}
    for group_name, entries in shared_tbme_group_suites().items():
        refs: list[SuiteRef] = []
        for entry in entries:
            env_preset_id = str(entry["env_preset_id"])
            refs.append(
                SuiteRef(
                    str(entry["suite_id"]),
                    _suite_label(env_preset_id),
                    session_root,
                    experiment_env_slug(env_preset_id),
                    tuple(str(policy_id) for policy_id in entry["policy_ids"]),
                )
            )
        groups_table[group_name] = refs
    return groups_table


def groups() -> dict[str, list[SuiteRef]]:
    global _groups
    if _groups is None:
        _groups = _build_groups(_results_dir)
    return _groups


def results_dir() -> Path:
    return _results_dir


def set_results_dir(new_results_dir: Path | str) -> None:
    global _results_dir, _groups
    _results_dir = Path(new_results_dir).resolve()
    _groups = _build_groups(_results_dir)


def session_root() -> Path:
    return latest_session(_results_dir)


def suite_dir(group_name: str, suite_id: str) -> Path:
    for ref in groups()[group_name]:
        if ref.suite_id == suite_id:
            return ref.session_root / "tracks" / ref.suite_id
    raise KeyError(f"Unknown suite {group_name}/{suite_id}")
