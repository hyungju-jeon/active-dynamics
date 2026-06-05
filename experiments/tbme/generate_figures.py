#!/usr/bin/env python3
from __future__ import annotations

import argparse

from experiments.tbme import tbme_figures


EXPERIMENT_GROUPS: dict[str, dict[str, tuple[str, ...]]] = {
    "exp01_base": {
        "plots": (
            "true_dynamics_all",
            "mismatch_dose_response",
            "sample_efficiency_thresholds",
            "compute_accuracy_pareto",
            "per_parameter_recovery",
            "information_learning_coupling",
        ),
    },
    "exp02_hard": {
        "plots": (
            "asymmetric_basin_mechanism",
            "learned_vectorfield_snapshots",
            "sample_efficiency_thresholds",
            "compute_accuracy_pareto",
        ),
    },
    "exp03_schedule": {
        "plots": ("compute_accuracy_pareto",),
    },
    "exp04_mismatch": {
        "plots": (
            "mismatch_dose_response",
            "sample_efficiency_thresholds",
            "compute_accuracy_pareto",
        ),
    },
    "exp05_ablation": {
        "plots": (
            "objective_ablation",
            "downstream_control",
        ),
    },
    "exp06_bottleneck": {
        "plots": ("bottleneck_sweep",),
    },
    "exp07_mismatch_stress": {
        "plots": ("mismatch_dose_response",),
    },
}

REQUESTED_PLOTS = {
    "bottleneck_sweep",
    "objective_ablation",
    "mismatch_dose_response",
    "downstream_control",
}
ADDITIONAL_PLOTS = {
    "true_dynamics_all",
    "asymmetric_basin_mechanism",
    "learned_vectorfield_snapshots",
    "sample_efficiency_thresholds",
    "compute_accuracy_pareto",
    "per_parameter_recovery",
    "information_learning_coupling",
}
DEFAULT_GROUPS = ",".join(EXPERIMENT_GROUPS)


def _summary_args(args: argparse.Namespace) -> list[str]:
    return ["--groups", str(args.groups), "--figure-formats", str(args.figure_formats)]


def _trajectory_args(args: argparse.Namespace, *, max_seeds: int | None = None) -> list[str]:
    return [
        "--groups",
        str(args.groups),
        "--max-seeds",
        str(args.max_seeds if max_seeds is None else max_seeds),
        "--density-bins",
        str(args.density_bins),
    ]


def _csv_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _experiment_plot_ids(group_csv: str) -> list[str]:
    group_ids = _csv_items(group_csv)
    unknown = sorted(set(group_ids) - set(EXPERIMENT_GROUPS))
    if unknown:
        raise ValueError(f"Unknown experiment group(s): {', '.join(unknown)}")

    plot_ids: list[str] = []
    seen: set[str] = set()
    for group_id in group_ids:
        for plot_id in EXPERIMENT_GROUPS[group_id]["plots"]:
            if plot_id not in seen:
                plot_ids.append(plot_id)
                seen.add(plot_id)
    return plot_ids


def _run_experiment_plots(args: argparse.Namespace) -> int:
    plot_ids = _experiment_plot_ids(str(args.groups))
    unknown = sorted(set(plot_ids) - REQUESTED_PLOTS - ADDITIONAL_PLOTS)
    if unknown:
        raise ValueError(f"Unknown experiment plot(s): {', '.join(unknown)}")

    requested_plot_ids = [plot_id for plot_id in plot_ids if plot_id in REQUESTED_PLOTS]
    additional_plot_ids = [plot_id for plot_id in plot_ids if plot_id in ADDITIONAL_PLOTS]

    if requested_plot_ids:
        code = int(tbme_figures.requested_main(["--plots", ",".join(requested_plot_ids)]))
        if code != 0:
            return code
    if additional_plot_ids:
        code = int(
            tbme_figures.additional_main(
                [
                    "--max-seeds",
                    str(args.max_seeds),
                    "--plots",
                    ",".join(additional_plot_ids),
                ]
            )
        )
        if code != 0:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME visual outputs.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="output_set")

    # Summary : Write summary figures from existing summary CSV files.
    summary = subparsers.add_parser(
        "summary",
        help="Write per-suite summary figures from existing summary CSV files.",
    )
    summary.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME group names.",
    )
    summary.add_argument("--figure-formats", type=str, default=".pdf")

    # Assets : Export current manuscript tables
    subparsers.add_parser(
        "assets",
        help="Export current manuscript tables and copied summary assets.",
    )

    # Trajectory : Generate trajectory overlay and density summary figures.
    trajectory = subparsers.add_parser(
        "trajectory",
        help="Generate trajectory overlay and density summary figures.",
    )
    trajectory.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME group names.",
    )
    trajectory.add_argument("--max-seeds", type=int, default=50)
    trajectory.add_argument("--density-bins", type=int, default=96)

    experiment = subparsers.add_parser(
        "experiment",
        help="Generate experiment-level manuscript figures.",
    )
    experiment.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME experiment group names.",
    )
    experiment.add_argument("--max-seeds", type=int, default=100)

    all_parser = subparsers.add_parser("all", help="Generate all TBME visual outputs.")
    all_parser.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME group names.",
    )
    all_parser.add_argument("--figure-formats", type=str, default=".pdf")
    all_parser.add_argument("--max-seeds", type=int, default=100)
    all_parser.add_argument("--trajectory-max-seeds", type=int, default=None)
    all_parser.add_argument("--density-bins", type=int, default=96)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_set is None:
        parser.print_help()
        return 0
    if args.output_set == "summary":
        return int(tbme_figures.summary_main(_summary_args(args)))
    if args.output_set == "assets":
        return int(tbme_figures.asset_main())
    if args.output_set == "trajectory":
        return int(tbme_figures.trajectory_main(_trajectory_args(args)))
    if args.output_set == "experiment":
        return _run_experiment_plots(args)
    if args.output_set == "all":
        trajectory_max_seeds = (
            args.trajectory_max_seeds if args.trajectory_max_seeds is not None else args.max_seeds
        )
        code = int(tbme_figures.summary_main(_summary_args(args)))
        if code != 0:
            return code
        code = int(tbme_figures.asset_main())
        if code != 0:
            return code
        code = int(
            tbme_figures.trajectory_main(
                _trajectory_args(args, max_seeds=int(trajectory_max_seeds))
            )
        )
        if code != 0:
            return code
        code = _run_experiment_plots(args)
        if code != 0:
            return code
        return 0
    raise ValueError(f"Unknown output set: {args.output_set}")


if __name__ == "__main__":
    raise SystemExit(main())
