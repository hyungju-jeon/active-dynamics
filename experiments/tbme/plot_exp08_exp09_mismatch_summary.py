#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ADAPTIVE = "adaptive"
FIXED = "active_planning"
GOOD_R2 = 0.9
EARLY_STEP_MAX = 500

GROUPS = [
    ("duffing", "mild"),
    ("duffing", "strong"),
    ("gated_duffing", "mild"),
    ("gated_duffing", "strong"),
]
FAMILIES = ["parameter", "observation"]



def _median(values: list[float]) -> float:
    values = [value for value in values if value == value]
    return statistics.median(values) if values else math.nan


def _quantile(values: list[float], q: float) -> float:
    values = np.asarray([value for value in values if value == value], dtype=float)
    return float(np.quantile(values, q)) if values.size else math.nan


def _mean(values: list[float]) -> float:
    values = [value for value in values if value == value]
    return sum(values) / len(values) if values else math.nan


def _parse_exp_id(exp_id: str) -> tuple[str, str, str] | None:
    for family, prefix, token in [
        ("parameter", "exp08_", "_parameter_mismatch_"),
        ("observation", "exp09_", "_observation_tuning_mismatch_"),
    ]:
        if not exp_id.startswith(prefix) or token not in exp_id:
            continue
        system, severity = exp_id.removeprefix(prefix).split(token, 1)
        if (system, severity) in GROUPS:
            return family, system, severity
    return None


def _read_run_metrics(run_dir: Path) -> tuple[float, float, float]:
    replan_count = 0
    with (run_dir / "information_trace.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row.get("adaptive_replan_reason") == "state_tracking_error":
                replan_count += 1

    early_r2: list[float] = []
    with (run_dir / "trajectory_r2_trace.csv").open() as handle:
        for row in csv.DictReader(handle):
            if float(row["step"]) <= EARLY_STEP_MAX:
                early_r2.append(float(row["trajectory_r2"]))

    good_r2_fraction = _mean([float(value >= GOOD_R2) for value in early_r2])
    return _mean(early_r2), good_r2_fraction, float(replan_count)


def _collect(root: Path, family: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for meta_path in root.glob("**/repeat_01/run_metadata.json"):
        metadata = json.loads(meta_path.read_text())
        policy = metadata.get("policy_id")
        if policy not in {ADAPTIVE, FIXED}:
            continue
        parsed = _parse_exp_id(str(metadata.get("exp_id", "")))
        if parsed is None or parsed[0] != family:
            continue
        info_path = meta_path.parent / "information_trace.csv"
        r2_path = meta_path.parent / "trajectory_r2_trace.csv"
        if not info_path.exists() or not r2_path.exists():
            continue
        early_r2, good_r2_fraction, replan_count = _read_run_metrics(meta_path.parent)
        rows.append(
            {
                "family": parsed[0],
                "system": parsed[1],
                "severity": parsed[2],
                "policy": policy,
                "seed": int(metadata["seed"]),
                "early_r2": early_r2,
                "good_r2_fraction": good_r2_fraction,
                "state_error_replans": replan_count,
            }
        )
    return rows


def _summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for family in FAMILIES:
        for system, severity in GROUPS:
            for policy in [FIXED, ADAPTIVE]:
                selected = [
                    row for row in rows
                    if row["family"] == family
                    and row["system"] == system
                    and row["severity"] == severity
                    and row["policy"] == policy
                ]
                for metric in ["early_r2", "good_r2_fraction", "state_error_replans"]:
                    values = [float(row[metric]) for row in selected]
                    out.append(
                        {
                            "family": family,
                            "system": system,
                            "severity": severity,
                            "policy": policy,
                            "metric": metric,
                            "n": len(values),
                            "median": _median(values),
                            "q25": _quantile(values, 0.25),
                            "q75": _quantile(values, 0.75),
                        }
                    )
    return out


def _stat(summary: list[dict[str, object]], family: str, system: str, severity: str, policy: str, metric: str) -> tuple[float, float, float]:
    for row in summary:
        if (
            row["family"] == family
            and row["system"] == system
            and row["severity"] == severity
            and row["policy"] == policy
            and row["metric"] == metric
        ):
            return float(row["median"]), float(row["q25"]), float(row["q75"])
    return math.nan, math.nan, math.nan


def _draw_bars(ax: plt.Axes, summary: list[dict[str, object]], family: str, metric: str, include_fixed: bool) -> None:
    colors = {FIXED: "#666666", ADAPTIVE: "#B44E5A"}
    policies = [FIXED, ADAPTIVE] if include_fixed else [ADAPTIVE]
    width = 0.34 if include_fixed else 0.52
    x = np.arange(len(GROUPS), dtype=float)
    for idx, policy in enumerate(policies):
        offset = (idx - 0.5) * width if include_fixed else 0.0
        medians: list[float] = []
        err_lo: list[float] = []
        err_hi: list[float] = []
        for system, severity in GROUPS:
            median, q25, q75 = _stat(summary, family, system, severity, policy, metric)
            medians.append(median)
            err_lo.append(median - q25)
            err_hi.append(q75 - median)
        ax.bar(
            x + offset,
            medians,
            width=width,
            yerr=[err_lo, err_hi],
            capsize=2,
            color=colors[policy],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.82,
            label="fixed" if policy == FIXED else "adaptive",
        )
    ax.set_xticks(x, ["Duffing\nmild", "Duffing\nstrong", "Basin\nmild", "Basin\nstrong"])
    ax.tick_params(axis="x", length=0)


def _write_csv(path: Path, summary: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def _plot(summary: list[dict[str, object]], output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(9.1, 5.2), constrained_layout=True)
    row_titles = {"parameter": "exp08 parameter mismatch", "observation": "exp09 observation-model mismatch"}
    metrics = [
        ("early_r2", "mean predictive $R^2$\nsteps <= 500", True, "early prediction"),
        ("good_r2_fraction", f"fraction early $R^2$ >= {GOOD_R2:g}", True, "good-performance occupancy"),
        ("state_error_replans", "state-error replans / run", False, "adaptive replanning"),
    ]
    letters = iter("ABCDEF")
    for row, family in enumerate(FAMILIES):
        for col, (metric, ylabel, include_fixed, title) in enumerate(metrics):
            ax = axes[row, col]
            _draw_bars(ax, summary, family, metric, include_fixed)
            ax.set_title(f"{next(letters)}. {row_titles[family]}: {title}")
            ax.set_ylabel(ylabel)
            if metric == "early_r2":
                ax.axhline(0.9, color="#888888", linestyle=":", linewidth=0.8)
                ax.set_ylim(-0.25, 1.02)
            if metric == "good_r2_fraction":
                ax.set_ylim(0.0, 1.02)
            if metric == "state_error_replans":
                ax.set_ylim(bottom=0.0)
    axes[0, 0].legend(frameon=False, loc="lower right")
    fig.suptitle("Mismatch stress tests: parameter mismatch versus observation-model mismatch", y=1.02, fontsize=11)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot exp08/exp09 mismatch comparison summary.")
    parser.add_argument(
        "--exp08-dir",
        type=Path,
        default=Path("results/tbme/gcp/tbme-gcp-n2-seed100-20260614-145731/results/exp08_parameter_mismatch_stress/session_1"),
    )
    parser.add_argument(
        "--exp09-dir",
        type=Path,
        default=Path("results/tbme/exp09_observation_tuning_mismatch/session_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tbme/mismatch_summary/exp08_exp09_mismatch_systematic"),
    )
    args = parser.parse_args()

    rows = _collect(args.exp08_dir, "parameter") + _collect(args.exp09_dir, "observation")
    summary = _summaries(rows)
    _write_csv(args.output.with_suffix(".csv"), summary)
    _plot(summary, args.output)
    print(args.output.with_suffix(".csv"))
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
