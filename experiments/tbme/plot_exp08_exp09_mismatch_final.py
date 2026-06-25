#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actdyn.utils.experiment_runtime import (
    read_trace_csv,
    safe_float as _safe_float,
    write_trace_csv,
)
from actdyn.utils.figure_io import (
    centered_moving_average,
    finite_mean,
    finite_median,
    finite_quantile,
)


ADAPTIVE = "adaptive"
FIXED = "active_planning"
SYSTEM = "gated_duffing"
THRESHOLD = 3.0
SHORT_HORIZON = 5
FIXED_REPLAN_INTERVAL = 20


def _parse_exp_id(exp_id: str) -> tuple[str, str] | None:
    families = [
        ("parameter", "exp08_gated_duffing_parameter_mismatch_"),
        ("observation", "exp09_gated_duffing_observation_tuning_mismatch_"),
    ]
    for family, prefix in families:
        if exp_id.startswith(prefix):
            severity = exp_id.removeprefix(prefix)
            if severity in {"mild", "strong"}:
                return family, severity
    return None


def _read_info(path: Path) -> tuple[list[dict[str, object]], dict[int, dict[str, object]]]:
    rows: list[dict[str, object]] = []
    by_step: dict[int, dict[str, object]] = {}
    for row in read_trace_csv(path):
        rec = {
            "step": int(float(row["step"])),
            "tracking_error": _safe_float(
                row.get("adaptive_state_tracking_error"), default=math.nan
            ),
            "replan_reason": str(row.get("adaptive_replan_reason", "none")),
            "parameter_updated": str(row.get("parameter_posterior_updated", "")).lower() == "true",
            "parameter_update_reason": str(row.get("parameter_update_reason", "none")),
            "I_theta_t": _safe_float(row.get("I_theta_t"), default=math.nan),
            "theta_block_eig": _safe_float(row.get("theta_block_eig"), default=math.nan),
        }
        rows.append(rec)
        by_step[int(rec["step"])] = rec
    return rows, by_step


def _read_r2(path: Path) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for row in read_trace_csv(path):
        step = int(float(row["step"]))
        if step <= 500:
            steps.append(step)
            values.append(float(row["trajectory_r2"]))
    return steps, values


def _collect(root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for meta_path in root.glob("**/repeat_01/run_metadata.json"):
        metadata = json.loads(meta_path.read_text())
        policy = metadata.get("policy_id")
        if policy not in {ADAPTIVE, FIXED}:
            continue
        parsed = _parse_exp_id(str(metadata.get("exp_id", "")))
        if parsed is None:
            continue
        info_path = meta_path.parent / "information_trace.csv"
        r2_path = meta_path.parent / "trajectory_r2_trace.csv"
        if not info_path.exists() or not r2_path.exists():
            continue
        info, info_by_step = _read_info(info_path)
        r2_steps, r2_values = _read_r2(r2_path)
        runs.append(
            {
                "family": parsed[0],
                "severity": parsed[1],
                "policy": policy,
                "seed": int(metadata["seed"]),
                "info": info,
                "info_by_step": info_by_step,
                "r2_steps": r2_steps,
                "r2_values": r2_values,
            }
        )
    return runs


def _clean_short_window(run: dict[str, object], event_step: int) -> bool:
    policy = str(run["policy"])
    if policy == FIXED and ((event_step - 1) % FIXED_REPLAN_INTERVAL) + SHORT_HORIZON >= FIXED_REPLAN_INTERVAL:
        return False
    by_step = run["info_by_step"]
    assert isinstance(by_step, dict)
    for step in range(event_step + 1, event_step + SHORT_HORIZON + 1):
        row = by_step.get(step)
        if row is None:
            return False
        if row["parameter_updated"]:
            return False
        if policy == ADAPTIVE and row["replan_reason"] not in {"none", ""}:
            return False
    return True


def _next_update(run: dict[str, object], event_step: int) -> tuple[float, str]:
    for row in run["info"]:
        if int(row["step"]) > event_step and row["parameter_updated"]:
            return float(int(row["step"]) - event_step), str(row["parameter_update_reason"])
    return math.nan, "none"


def _event_record(family: str, label: str, run: dict[str, object], row: dict[str, object]) -> dict[str, object] | None:
    event_step = int(row["step"])
    by_step = run["info_by_step"]
    assert isinstance(by_step, dict)
    post = by_step.get(event_step + SHORT_HORIZON)
    if post is None:
        return None
    delay, update_reason = _next_update(run, event_step)
    return {
        "family": family,
        "label": label,
        "seed": int(run["seed"]),
        "event_step": event_step,
        "tracking_error_event": float(row["tracking_error"]),
        "short_window_clean": _clean_short_window(run, event_step),
        "tracking_error_plus5": float(post["tracking_error"]),
        "next_update_delay": delay,
        "next_update_reason": update_reason,
    }


def _matched_events(runs: list[dict[str, object]], family: str) -> list[dict[str, object]]:
    by_seed: dict[int, dict[str, dict[str, object]]] = defaultdict(dict)
    for run in runs:
        if run["family"] == family and run["severity"] == "strong":
            by_seed[int(run["seed"])][str(run["policy"])] = run

    records: list[dict[str, object]] = []
    for _, policies in by_seed.items():
        if ADAPTIVE not in policies:
            continue
        adaptive = policies[ADAPTIVE]
        adaptive_events = [
            row
            for row in adaptive["info"]
            if row["replan_reason"] == "state_tracking_error"
            and row["tracking_error"] == row["tracking_error"]
            and int(row["step"]) <= 1700
        ]
        fixed = policies.get(FIXED)
        fixed_candidates: list[dict[str, object]] = []
        if fixed is not None:
            fixed_candidates = [
                row
                for row in fixed["info"]
                if row["tracking_error"] == row["tracking_error"]
                and float(row["tracking_error"]) > THRESHOLD
                and int(row["step"]) <= 1700
            ]
        used_fixed_steps: set[int] = set()
        for event in adaptive_events:
            adaptive_record = _event_record(family, f"{family} adaptive", adaptive, event)
            if adaptive_record is not None:
                records.append(adaptive_record)
            if fixed is None or not fixed_candidates:
                continue
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
            used_fixed_steps.add(int(fixed_event["step"]))
            fixed_record = _event_record(family, f"{family} fixed", fixed, fixed_event)
            if fixed_record is not None:
                records.append(fixed_record)
    return records


def _plot_stale_burden(ax: plt.Axes, runs: list[dict[str, object]], colors: dict[str, str]) -> None:
    specs = [
        ("observation", "mild", ADAPTIVE, "obs adaptive, mild", colors["mild"], "-"),
        ("observation", "strong", ADAPTIVE, "obs adaptive, strong", colors["strong"], "-"),
        ("parameter", "mild", ADAPTIVE, "param adaptive, mild", colors["mild"], "--"),
        ("parameter", "strong", ADAPTIVE, "param adaptive, strong", colors["strong"], "--"),
        ("observation", "strong", FIXED, "obs fixed, strong", colors["fixed"], ":"),
        ("parameter", "strong", FIXED, "param fixed, strong", colors["fixed"], "-."),
    ]
    for family, severity, policy, label, color, linestyle in specs:
        by_step: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            if run["family"] == family and run["severity"] == severity and run["policy"] == policy:
                for row in run["info"]:
                    value = float(row["tracking_error"])
                    if value == value:
                        by_step[int(row["step"])].append(float(value > THRESHOLD))
        if not by_step:
            continue
        steps = np.asarray(sorted(by_step), dtype=float)
        frac = np.asarray([finite_mean(by_step[int(step)]) for step in steps], dtype=float)
        ax.plot(
            steps,
            centered_moving_average(frac),
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=label,
        )
    ax.set_title("A. mismatch creates stale plans")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("fraction above\ntracking threshold")
    ax.set_ylim(0.0, 0.13)
    ax.legend(frameon=False, loc="upper right", ncol=2)


def _plot_event_correction(ax: plt.Axes, events: list[dict[str, object]], colors: dict[str, str]) -> None:
    order = ["observation adaptive", "observation fixed", "parameter adaptive", "parameter fixed"]
    positions = [0, 1, 3, 4, 6, 7, 9, 10]
    box_data: list[list[float]] = []
    patch_colors: list[str] = []
    for label in order:
        clean = [row for row in events if row["label"] == label and row["short_window_clean"]]
        box_data.append([float(row["tracking_error_event"]) for row in clean])
        box_data.append([float(row["tracking_error_plus5"]) for row in clean])
        patch_colors.extend([colors[label], colors[label]])
    bp = ax.boxplot(box_data, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
    for patch, color in zip(bp["boxes"], patch_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    for line in bp["medians"]:
        line.set_color("black")
    ax.axhline(THRESHOLD, color="#333333", linewidth=0.9, linestyle=":")
    ax.set_yscale("log")
    ax.set_xticks(positions, ["event", "+5", "event", "+5", "event", "+5", "event", "+5"])
    ax.set_ylabel("tracking error\n(log scale)")
    ax.set_title("B. replanning clears stale plans")
    ax.text(0.16, 0.95, "obs adaptive", transform=ax.transAxes, color=colors["observation adaptive"], ha="center", va="top")
    ax.text(0.39, 0.95, "obs fixed", transform=ax.transAxes, color=colors["observation fixed"], ha="center", va="top")
    ax.text(0.62, 0.95, "param adaptive", transform=ax.transAxes, color=colors["parameter adaptive"], ha="center", va="top")
    ax.text(0.86, 0.95, "param fixed", transform=ax.transAxes, color=colors["parameter fixed"], ha="center", va="top")


def _plot_update_delay(ax: plt.Axes, events: list[dict[str, object]], colors: dict[str, str]) -> None:
    labels = ["observation adaptive", "observation fixed", "parameter adaptive", "parameter fixed"]
    for idx, label in enumerate(labels):
        vals = [float(row["next_update_delay"]) for row in events if row["label"] == label]
        q1, med, q3 = finite_quantile(vals, 0.25), finite_median(vals), finite_quantile(vals, 0.75)
        ax.bar(idx, med, color=colors[label], edgecolor="black", linewidth=0.4, width=0.6)
        ax.plot([idx, idx], [q1, q3], color="black", linewidth=1.2)
        if med == med:
            ax.text(idx, med + 0.45, f"{med:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(labels)), ["obs\nadaptive", "obs\nfixed", "param\nadaptive", "param\nfixed"])
    ax.set_ylabel("steps to next\nparameter update")
    ax.set_title("C. update arrives sooner")


def _plot_update_trigger(ax: plt.Axes, events: list[dict[str, object]], colors: dict[str, str]) -> None:
    labels = ["observation adaptive", "observation fixed", "parameter adaptive", "parameter fixed"]
    values = [
        finite_mean(
            [float(row["next_update_reason"] == "block_eig") for row in events if row["label"] == label]
        )
        for label in labels
    ]
    ax.bar(range(len(labels)), values, color=[colors[label] for label in labels], edgecolor="black", linewidth=0.4, width=0.65)
    ax.set_xticks(range(len(labels)), ["obs\nadaptive", "obs\nfixed", "param\nadaptive", "param\nfixed"])
    ax.set_ylim(0.0, 0.75)
    ax.set_ylabel("fraction block-EIG\nat next update")
    ax.set_title("D. update is information-triggered")
    for idx, value in enumerate(values):
        if value == value:
            ax.text(idx, value + 0.025, f"{value:.0%}", ha="center", fontsize=8)


def _plot_early_r2(ax: plt.Axes, runs: list[dict[str, object]], colors: dict[str, str]) -> None:
    specs = [
        ("observation", ADAPTIVE, "obs adaptive", colors["adaptive"], "-"),
        ("observation", FIXED, "obs fixed", colors["fixed"], "-"),
        ("parameter", ADAPTIVE, "param adaptive", colors["adaptive"], "--"),
        ("parameter", FIXED, "param fixed", colors["fixed"], "--"),
    ]
    for family, policy, label, color, linestyle in specs:
        by_step: dict[int, list[float]] = defaultdict(list)
        for run in runs:
            if run["family"] == family and run["severity"] == "strong" and run["policy"] == policy:
                for step, value in zip(run["r2_steps"], run["r2_values"]):
                    by_step[int(step)].append(float(value))
        if not by_step:
            continue
        steps = np.asarray(sorted(by_step), dtype=float)
        med = np.asarray([finite_median(by_step[int(step)]) for step in steps], dtype=float)
        lo = np.asarray([finite_quantile(by_step[int(step)], 0.25) for step in steps], dtype=float)
        hi = np.asarray([finite_quantile(by_step[int(step)], 0.75) for step in steps], dtype=float)
        ax.plot(steps, med, color=color, linestyle=linestyle, linewidth=1.7, label=label)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.9, color="#888888", linewidth=0.8, linestyle=":")
    ax.set_xlim(0, 500)
    ax.set_ylim(-0.7, 1.02)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("predictive $R^2$")
    ax.set_title("E. early prediction under strong mismatch")
    ax.legend(frameon=False, loc="lower right")


def _write_events(path: Path, events: list[dict[str, object]]) -> None:
    fields = [
        "family",
        "label",
        "seed",
        "event_step",
        "tracking_error_event",
        "short_window_clean",
        "tracking_error_plus5",
        "next_update_delay",
        "next_update_reason",
    ]
    write_trace_csv(path, events, fields)


def _plot(runs: list[dict[str, object]], events: list[dict[str, object]], output_base: Path) -> None:
    colors = {
        "adaptive": "#B44E5A",
        "fixed": "#666666",
        "mild": "#F58518",
        "strong": "#B44E5A",
        "observation adaptive": "#B44E5A",
        "observation fixed": "#666666",
        "parameter adaptive": "#4C78A8",
        "parameter fixed": "#888888",
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
    fig = plt.figure(figsize=(10.8, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.0])
    axes = [
        fig.add_subplot(gs[0, 0:3]),
        fig.add_subplot(gs[0, 3:6]),
        fig.add_subplot(gs[1, 0:2]),
        fig.add_subplot(gs[1, 2:4]),
        fig.add_subplot(gs[1, 4:6]),
    ]
    _plot_stale_burden(axes[0], runs, colors)
    _plot_event_correction(axes[1], events, colors)
    _plot_update_delay(axes[2], events, colors)
    _plot_update_trigger(axes[3], events, colors)
    _plot_early_r2(axes[4], runs, colors)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot final mismatch mechanism figure with exp08 parameter results.")
    parser.add_argument(
        "--exp09-dir",
        type=Path,
        default=Path("results/tbme/exp09_observation_tuning_mismatch/session_1"),
    )
    parser.add_argument(
        "--exp08-dir",
        type=Path,
        default=Path("results/tbme/exp08_parameter_mismatch_stress"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/tbme/exp09_observation_tuning_mismatch/session_1/summary/adaptive_mismatch_mechanism_final_with_parameter_gated_duffing"),
    )
    args = parser.parse_args()

    exp08_sessions = sorted(
        (int(path.name.removeprefix("session_")), path)
        for path in args.exp08_dir.glob("session_*")
        if path.name.removeprefix("session_").isdigit()
    )
    exp08_dir = exp08_sessions[-1][1] if exp08_sessions else args.exp08_dir
    runs = _collect(args.exp09_dir) + _collect(exp08_dir)
    events = _matched_events(runs, "observation") + _matched_events(runs, "parameter")
    _write_events(args.output.with_suffix(".events.csv"), events)
    _plot(runs, events, args.output)
    print(args.output.with_suffix(".events.csv"))
    print(args.output.with_suffix(".png"))
    print(args.output.with_suffix(".pdf"))
    for label in ["observation adaptive", "observation fixed", "parameter adaptive", "parameter fixed"]:
        total = sum(row["label"] == label for row in events)
        clean = sum(row["label"] == label and row["short_window_clean"] for row in events)
        print(label, "events", total, "clean", clean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
