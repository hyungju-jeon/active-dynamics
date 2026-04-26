#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "figs" / "tbme" / "generated"
TEX_DIR = REPO_ROOT / "docs" / "active-dynamics-writing" / "generated"


def _latest_session(base: Path) -> Path:
    sessions = [
        path for path in base.glob("session_*")
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


GROUPS: dict[str, list[SuiteRef]] = {
    "exp1_main": [
        SuiteRef("tbme_exp1_duffing_main", "Duffing", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1"), "duffing"),
        SuiteRef("tbme_exp1_damped_pendulum_main", "Damped pendulum", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1"), "damped_pendulum"),
        SuiteRef("tbme_exp1_asymmetric_basin_main", "Asymmetric basin", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1"), "asymmetric_basin"),
        SuiteRef("tbme_exp1_multi_stable_main", "Multi-stable", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1"), "multi_stable"),
    ],
    "exp1_schedule": [
        SuiteRef("tbme_exp1_schedule_duffing", "Duffing", REPO_ROOT / "results" / "tbme" / "exp1_1_with_flex_official_updated" / "session_1", "duffing"),
        SuiteRef("tbme_exp1_schedule_damped_pendulum", "Damped pendulum", REPO_ROOT / "results" / "tbme" / "exp1_1_with_flex_official_updated" / "session_1", "damped_pendulum"),
        SuiteRef("tbme_exp1_schedule_asymmetric_basin", "Asymmetric basin", REPO_ROOT / "results" / "tbme" / "exp1_1_with_flex_official_updated" / "session_1", "asymmetric_basin"),
        SuiteRef("tbme_exp1_schedule_multi_stable", "Multi-stable", REPO_ROOT / "results" / "tbme" / "exp1_1_with_flex_official_updated" / "session_1", "multi_stable"),
    ],
    "exp1_hard": [
        SuiteRef("tbme_exp1_hard_duffing_asymmetric", "Duffing asymmetric obs", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1_2_hard"), "duffing_asymmetric"),
        SuiteRef("tbme_exp1_hard_duffing", "Duffing hard", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1_2_hard"), "duffing_hard"),
        SuiteRef("tbme_exp1_hard_damped_pendulum_asymmetry", "Pendulum asymmetric obs", _latest_session(REPO_ROOT / "results" / "tbme" / "exp1_2_hard"), "damped_pendulum_asymmetry"),
    ],
    "exp2": [
        SuiteRef("tbme_exp2_duffing_parameter_mismatch", "Duffing parameter mismatch", REPO_ROOT / "results" / "tbme" / "exp2_with_flex_official_updated" / "session_1", "duffing_parameter_mismatch"),
        SuiteRef("tbme_exp2_asymmetric_basin_parameter_mismatch", "Asymmetric basin parameter mismatch", REPO_ROOT / "results" / "tbme" / "exp2_with_flex_official_updated" / "session_1", "asymmetric_basin_parameter_mismatch"),
    ],
}

POLICY_LABELS = {
    "active_planning": "Planning",
    "active_planning_u1_r1_h40": "Planning u1/r1/h40",
    "active_planning_u5_r5_h40": "Planning u5/r5/h40",
    "active_planning_u1_r5_h40": "Planning u1/r5/h40",
    "active_planning_u10_r10_h40": "Planning u10/r10/h40",
    "active_planning_u5_r10_h40": "Planning u5/r10/h40",
    "active_myopic": "Myopic",
    "prbs": "PRBS",
    "random": "Random",
    "flex": "FLEX",
    "flex_official": "FLEX (official)",
    "ensemble": "Ensemble",
    "rhc": "RHC-US",
}

POLICY_ORDER = [
    "active_planning",
    "active_planning_u1_r1_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u1_r5_h40",
    "active_planning_u10_r10_h40",
    "active_planning_u5_r10_h40",
    "active_myopic",
    "prbs",
    "random",
    "flex",
    "flex_official",
    "ensemble",
    "rhc",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _summary_dir(ref: SuiteRef) -> Path:
    return ref.session_root / ref.suite_id / 'summary'


def _policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        idx = POLICY_ORDER.index(policy_id)
    except ValueError:
        idx = len(POLICY_ORDER)
    return idx, policy_id


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return '--'
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _aggregate_suite(ref: SuiteRef) -> list[dict[str, object]]:
    rows = _read_csv(_summary_dir(ref) / 'metrics.csv')
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row.get('status') != 'completed':
            continue
        pid = str(row['policy_id'])
        bucket = grouped.setdefault(pid, {'value': [], 'r2': [], 'runtime': []})
        if row.get('value_final_mean'):
            bucket['value'].append(float(row['value_final_mean']))
        if row.get('trajectory_r2_final_mean'):
            bucket['r2'].append(float(row['trajectory_r2_final_mean']))
        if row.get('runtime_sec_mean'):
            bucket['runtime'].append(float(row['runtime_sec_mean']))
    out: list[dict[str, object]] = []
    for pid, bucket in grouped.items():
        vals = np.asarray(bucket['value'], dtype=np.float64)
        r2s = np.asarray(bucket['r2'], dtype=np.float64)
        runtimes = np.asarray(bucket['runtime'], dtype=np.float64)
        out.append({
            'suite_id': ref.suite_id,
            'suite_label': ref.label,
            'policy_id': pid,
            'policy_label': POLICY_LABELS.get(pid, pid),
            'n': int(vals.size),
            'parameter_error_mean': float(vals.mean()) if vals.size else math.nan,
            'parameter_error_std': float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            'trajectory_r2_mean': float(r2s.mean()) if r2s.size else math.nan,
            'trajectory_r2_std': float(r2s.std(ddof=1)) if r2s.size > 1 else 0.0,
            'runtime_sec_mean': float(runtimes.mean()) if runtimes.size else math.nan,
            'runtime_sec_std': float(runtimes.std(ddof=1)) if runtimes.size > 1 else 0.0,
        })
    out.sort(key=lambda row: _policy_sort_key(str(row['policy_id'])))
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'suite_id','suite_label','policy_id','policy_label','n',
        'parameter_error_mean','parameter_error_std',
        'trajectory_r2_mean','trajectory_r2_std',
        'runtime_sec_mean','runtime_sec_std',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _escape(text: str) -> str:
    return text.replace('_', r'\_')


def _write_tex(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    lines = [
        '% Auto-generated by experiments/tbme/export_current_tbme_assets.py',
        r'\begin{tabular}{llccc}',
        r'\toprule',
        r'Suite & Policy & Param. error & Trajectory $R^2$ & Runtime (s) \\',
        r'\midrule',
    ]
    current_suite = None
    for row in rows:
        suite = str(row['suite_label'])
        suite_cell = _escape(suite) if suite != current_suite else ''
        current_suite = suite
        line = " & ".join([
            suite_cell,
            _escape(str(row['policy_label'])),
            _fmt(float(row['parameter_error_mean']), float(row['parameter_error_std'])),
            _fmt(float(row['trajectory_r2_mean']), float(row['trajectory_r2_std'])),
            _fmt(float(row['runtime_sec_mean']), float(row['runtime_sec_std']), digits=1),
        ]) + r" \\"
        lines.append(line)
    lines += [r'\bottomrule', r'\end{tabular}']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _clear_group_figures(group_name: str) -> None:
    group_dir = FIG_DIR / group_name
    if group_dir.exists():
        shutil.rmtree(group_dir)
    for legacy in FIG_DIR.glob(f"{group_name}_*.pdf"):
        legacy.unlink()


def _copy_figures(group_name: str, refs: list[SuiteRef]) -> list[Path]:
    copied: list[Path] = []
    _clear_group_figures(group_name)
    for ref in refs:
        fig_dir = _summary_dir(ref) / 'figures'
        if not fig_dir.exists():
            continue
        dst_dir = FIG_DIR / group_name / ref.slug
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(fig_dir.glob('*.pdf')):
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def export_group(group_name: str, refs: list[SuiteRef]) -> tuple[list[dict[str, object]], list[Path]]:
    rows: list[dict[str, object]] = []
    for ref in refs:
        rows.extend(_aggregate_suite(ref))
    rows.sort(key=lambda row: (str(row['suite_label']), _policy_sort_key(str(row['policy_id']))))
    csv_path = TEX_DIR / f"tbme_{group_name}_table.csv"
    tex_path = TEX_DIR / f"tbme_{group_name}_table.tex"
    _write_csv(csv_path, rows)
    _write_tex(tex_path, group_name, rows)
    copied = _copy_figures(group_name, refs)
    return rows, copied


def main() -> int:
    summary_lines = []
    for group_name, refs in GROUPS.items():
        rows, copied = export_group(group_name, refs)
        summary_lines.append(f"{group_name}: {len(rows)} table rows, {len(copied)} copied figures")
    manifest = TEX_DIR / 'tbme_current_export_manifest.txt'
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print('\n'.join(summary_lines))
    print(manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
