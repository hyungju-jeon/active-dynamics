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

from actdyn.utils.figure_io import (
    centered_moving_average,
    finite_mean,
    finite_median,
    finite_quantile,
)
from experiments.tbme.tbme_io import safe_float as _safe_float


ADAPTIVE = "active_planning_adaptive_u20_r20_h40"
FIXED = "active_planning_u20_r20_h40"
SYSTEM = "asymmetric_basin"
THRESHOLD = 3.0
SHORT_HORIZON = 5
FIXED_REPLAN_INTERVAL = 20


def _sem(values: list[float]) -> float:
    values = [value for value in values if value == value]
    if len(values) < 2:
        return math.nan
    return statistics.stdev(values) / math.sqrt(len(values))


def _condition_from_exp_id(exp_id: str) -> tuple[str, str] | None:
    if exp_id == "exp01_asymmetric_basin":
        return SYSTEM, "perfect"
    prefix = "exp09_asymmetric_basin_observation_tuning_mismatch_"
    if exp_id.startswith(prefix):
        return SYSTEM, exp_id.removeprefix(prefix)
    return None


def _read_info_trace(path: Path) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    by_step: dict[int, dict[str, object]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rec = {
                "step": int(float(row["step"])),
                "tracking_error": _safe_float(
                    row.get("adaptive_state_tracking_error"), default=math.nan
                ),
                "replan_reason": str(row.get("adaptive_replan_reason", "none")),
                "parameter_updated": str(row.get("parameter_posterior_updated", "")).lower()
                == "true",
            }
            rows.append(rec)
            by_step[int(rec["step"])] = rec
    return rows, by_step


def _read_r2_trace(path: Path) -> tuple[list[float], list[float]]:
    steps: list[float] = []
    values: list[float] = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            steps.append(float(row["step"]))
            values.append(float(row["trajectory_r2"]))
    return steps, values


def _collect_runs(result_dir: Path, perfect_dir: Path | None) -> list[dict[str, object]]:
    """Load trace data needed for the compact mismatch-mechanism figure."""
    roots = [result_dir]
    if perfect_dir is not None and perfect_dir.exists():
        roots.append(perfect_dir)

    runs: list[dict[str, object]] = []
    for root in roots:
        for meta_path in root.glob("**/repeat_01/run_metadata.json"):
            metadata = json.loads(meta_path.read_text())
            policy = metadata.get("policy_id")
            if policy not in {ADAPTIVE, FIXED}:
                continue
            parsed = _condition_from_exp_id(str(metadata.get("exp_id", "")))
            if parsed is None:
                continue
            system, condition = parsed
            if system != SYSTEM:
                continue

            info_path = meta_path.parent / "information_trace.csv"
            r2_path = meta_path.parent / "trajectory_r2_trace.csv"
            if not info_path.exists() or not r2_path.exists():
                continue
            info_rows, info_by_step = _read_info_trace(info_path)
            r2_steps, r2_values = _read_r2_trace(r2_path)
            runs.append(
                {
                    "condition": condition,
                    "policy": policy,
                    "seed": int(metadata["seed"]),
                    "info": info_rows,
                    "info_by_step": info_by_step,
                    "r2_steps": r2_steps,
                    "r2_values": r2_values,
                }
            )
    return runs


def _clean_short_window(run: dict[str, object], event_step: int) -> bool:
    info_by_step = run["info_by_step"]
    assert isinstance(info_by_step, dict)
    policy = run["policy"]
    if policy == FIXED and ((event_step - 1) % FIXED_REPLAN_INTERVAL) + SHORT_HORIZON >= FIXED_REPLAN_INTERVAL:
        return False
    for step in range(event_step + 1, event_step + SHORT_HORIZON + 1):
        row = info_by_step.get(step)
        if row and row["parameter_updated"]:
            return False
        if policy == ADAPTIVE and row and row["replan_reason"] not in {"none", ""}:
            return False
    return True


def _match_strong_events(runs: list[dict[str, object]]) -> list[dict[str, float]]:
    """Match adaptive state-error replans to fixed high-error windows by seed and phase."""
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for run in runs:
        if run["condition"] == "strong":
            by_seed[int(run["seed"])][str(run["policy"])] = run

    matches: list[dict[str, float]] = []
    for seed, policies in by_seed.items():
        if ADAPTIVE not in policies or FIXED not in policies:
            continue
        adaptive = policies[ADAPTIVE]
        fixed = policies[FIXED]
        adaptive_events = [
            row
            for row in adaptive["info"]
            if row["replan_reason"] == "state_tracking_error"
            and row["tracking_error"] == row["tracking_error"]
            and int(row["step"]) <= 1700
        ]
        fixed_candidates = [
            row
            for row in fixed["info"]
            if row["tracking_error"] == row["tracking_error"]
            and float(row["tracking_error"]) > THRESHOLD
            and int(row["step"]) <= 1700
        ]
        used_fixed_steps: set[int] = set()
        for event in adaptive_events:
            event_step = int(event["step"])
            event_error = float(event["tracking_error"])
            choices = [
                row
                for row in fixed_candidates
                if int(row["step"]) not in used_fixed_steps and abs(int(row["step"]) - event_step) <= 200
            ]
            if not choices:
                continue
            fixed_event = min(
                choices,
                key=lambda row: abs(int(row["step"]) - event_step) / 100.0
                + abs(math.log1p(float(row["tracking_error"])) - math.log1p(event_error)),
            )
            fixed_step = int(fixed_event["step"])
            used_fixed_steps.add(fixed_step)
            if not _clean_short_window(adaptive, event_step) or not _clean_short_window(fixed, fixed_step):
                continue

            adaptive_by_step = adaptive["info_by_step"]
            fixed_by_step = fixed["info_by_step"]
            assert isinstance(adaptive_by_step, dict)
            assert isinstance(fixed_by_step, dict)
            matches.append(
                {
                    "adaptive_pre": event_error,
                    "adaptive_post": float(adaptive_by_step[event_step + SHORT_HORIZON]["tracking_error"]),
                    "fixed_pre": float(fixed_event["tracking_error"]),
                    "fixed_post": float(fixed_by_step[fixed_step + SHORT_HORIZON]["tracking_error"]),
                }
            )
    return matches


def _plot_compact_figure(runs: list[dict[str, object]], output_base: Path) -> None:
    matches = _match_strong_events(runs)
    if not matches:
        raise RuntimeError("No clean matched replan events found.")

    colors = {
        "perfect": "#4C78A8",
        "mild": "#F58518",
        "strong": "#B44E5A",
        "adaptive": "#B44E5A",
        "fixed": "#666666",
    }
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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.reshape(-1)

    # A. How often plans become stale.
    for condition, policy, label, color, linestyle in [
        ("perfect", ADAPTIVE, "adaptive, perfect", colors["perfect"], "-"),
        ("mild", ADAPTIVE, "adaptive, mild", colors["mild"], "-"),
        ("strong", ADAPTIVE, "adaptive, strong", colors["strong"], "-"),
        ("strong", FIXED, "fixed, strong", colors["fixed"], "--"),
    ]:
        by_step: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            if run["condition"] == condition and run["policy"] == policy:
                for row in run["info"]:
                    value = float(row["tracking_error"])
                    if value == value:
                        by_step[int(row["step"])].append(float(value > THRESHOLD))
        if not by_step:
            continue
        steps = np.asarray(sorted(by_step), dtype=float)
        frac = np.asarray([finite_mean(by_step[int(step)]) for step in steps], dtype=float)
        ax_a.plot(
            steps,
            centered_moving_average(frac),
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )
    ax_a.set_title("A. mismatch makes plans stale")
    ax_a.set_xlabel("rollout step")
    ax_a.set_ylabel("fraction above\ntracking threshold")
    ax_a.set_ylim(0.0, 0.12)
    ax_a.legend(frameon=False, loc="upper right")

    # B. What a replan does immediately.
    box_data = [
        [row["adaptive_pre"] for row in matches],
        [row["adaptive_post"] for row in matches],
        [row["fixed_pre"] for row in matches],
        [row["fixed_post"] for row in matches],
    ]
    positions = [0, 1, 3, 4]
    bp = ax_b.boxplot(box_data, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], [colors["adaptive"], colors["adaptive"], colors["fixed"], colors["fixed"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for line in bp["medians"]:
        line.set_color("black")
    ax_b.axhline(THRESHOLD, color="#333333", linewidth=0.9, linestyle=":")
    ax_b.set_yscale("log")
    ax_b.set_xticks(positions, ["event", "+5", "event", "+5"])
    ax_b.set_ylabel("tracking error\n(log scale)")
    ax_b.set_title("B. replanning clears stale plans")
    ax_b.text(0.25, 0.96, "adaptive", transform=ax_b.transAxes, color=colors["adaptive"], ha="center", va="top")
    ax_b.text(0.75, 0.96, "fixed", transform=ax_b.transAxes, color=colors["fixed"], ha="center", va="top")

    # C. Mismatch increases how often adaptive replans.
    conditions = ["perfect", "mild", "strong"]
    labels = ["perfect", "mild\nmismatch", "strong\nmismatch"]
    means: list[float] = []
    sems: list[float] = []
    for condition in conditions:
        counts: list[float] = []
        for run in runs:
            if run["condition"] == condition and run["policy"] == ADAPTIVE:
                counts.append(
                    float(
                        sum(
                            row["replan_reason"] == "state_tracking_error"
                            for row in run["info"]
                        )
                    )
                )
        means.append(finite_mean(counts))
        sems.append(_sem(counts))
    x = np.arange(len(conditions), dtype=float)
    ax_c.bar(
        x,
        means,
        yerr=sems,
        color=[colors["perfect"], colors["mild"], colors["strong"]],
        edgecolor="black",
        linewidth=0.4,
        capsize=2,
        width=0.62,
    )
    ax_c.set_xticks(x, labels)
    ax_c.set_ylabel("state-error replans / run")
    ax_c.set_title("C. stronger mismatch requests\nmore replanning")
    for xi, value in zip(x, means):
        ax_c.text(xi, value + 0.18, f"{value:.1f}", ha="center", fontsize=8)

    # D. Outcome: early predictive R2 under strong mismatch.
    for policy, label, color in [(ADAPTIVE, "adaptive", colors["adaptive"]), (FIXED, "fixed", colors["fixed"])]:
        by_step: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            if run["condition"] == "strong" and run["policy"] == policy:
                for step, value in zip(run["r2_steps"], run["r2_values"]):
                    if step <= 500:
                        by_step[int(step)].append(float(value))
        steps = np.asarray(sorted(by_step), dtype=float)
        med = np.asarray([finite_median(by_step[int(step)]) for step in steps], dtype=float)
        lo = np.asarray([finite_quantile(by_step[int(step)], 0.25) for step in steps], dtype=float)
        hi = np.asarray([finite_quantile(by_step[int(step)], 0.75) for step in steps], dtype=float)
        ax_d.plot(steps, med, color=color, linewidth=1.7, label=label)
        ax_d.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax_d.axhline(0.9, color="#888888", linewidth=0.8, linestyle=":")
    ax_d.set_xlim(0, 500)
    ax_d.set_ylim(-0.2, 1.02)
    ax_d.set_xlabel("rollout step")
    ax_d.set_ylabel("predictive $R^2$")
    ax_d.set_title("D. early prediction improves modestly")
    ax_d.legend(frameon=False, loc="lower right")

    fig.suptitle("Adaptive planning under observation-model mismatch", y=1.02, fontsize=11)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the compact exp09 mismatch mechanism figure.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("results/tbme/exp09_observation_tuning_mismatch/session_1"),
    )
    parser.add_argument(
        "--perfect-dir",
        type=Path,
        default=Path("results/tbme/gcp/tbme-gcp-n2-seed100-20260614-145731/results/exp01_base/session_1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tbme/exp09_observation_tuning_mismatch/session_1/summary/adaptive_mismatch_mechanism_compact_asymmetric_basin"),
    )
    args = parser.parse_args()

    runs = _collect_runs(args.result_dir, args.perfect_dir)
    _plot_compact_figure(runs, args.output)
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
