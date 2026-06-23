#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.tbme import tbme_figures


EXPERIMENT_PLOTS_BY_GROUP: dict[str, tuple[str, ...]] = {
    "simple_system_identification": (
        "true_dynamics_all",
        "mismatch_dose_response",
        "per_parameter_recovery",
    ),
    "observation_action_bottleneck": ("bottleneck_sweep",),
    "model_mismatch": ("mismatch_dose_response",),
    "objective_ablation": ("objective_ablation",),
    "scheduling": (),
}

DEFAULT_GROUPS = ",".join(tbme_figures.GROUPS)


def _summary_args(args: argparse.Namespace) -> list[str]:
    return [
        "--groups",
        str(args.groups),
        "--figure-formats",
        str(args.figure_formats),
        "--trajectory-max-seeds",
        str(args.trajectory_max_seeds),
        "--density-bins",
        str(args.density_bins),
    ]


def _overview_args(args: argparse.Namespace) -> list[str]:
    return ["--groups", str(args.groups)]


def _csv_items(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _experiment_plot_ids(group_csv: str) -> list[str]:
    group_ids = _csv_items(group_csv)
    unknown = sorted(set(group_ids) - set(tbme_figures.GROUPS))
    if unknown:
        raise ValueError(f"Unknown experiment group(s): {', '.join(unknown)}")
    missing = sorted(set(group_ids) - set(EXPERIMENT_PLOTS_BY_GROUP))
    if missing:
        raise ValueError(f"No experiment plots configured for group(s): {', '.join(missing)}")

    plot_ids: list[str] = []
    seen: set[str] = set()
    for group_id in group_ids:
        for plot_id in EXPERIMENT_PLOTS_BY_GROUP[group_id]:
            if plot_id not in seen:
                plot_ids.append(plot_id)
                seen.add(plot_id)
    return plot_ids


def _experiment_args(args: argparse.Namespace) -> list[str]:
    plot_ids = _experiment_plot_ids(str(args.groups))
    unknown = sorted(set(plot_ids) - set(tbme_figures.EXPERIMENT_PLOTS))
    if unknown:
        raise ValueError(f"Unknown experiment plot(s): {', '.join(unknown)}")
    return [
        "--max-seeds",
        str(args.max_seeds),
        "--plots",
        ",".join(plot_ids),
        "--groups",
        str(args.groups),
    ]


def _assets_args(args: argparse.Namespace) -> list[str]:
    argv = ["--groups", str(args.groups)]
    results_dir = getattr(args, "results_dir", None)
    if results_dir is not None:
        argv.extend(["--results-dir", str(results_dir)])
    output_dir = getattr(args, "output_dir", None)
    if output_dir is None:
        output_dir = getattr(args, "assets_output_dir", None)
    if output_dir is not None:
        argv.extend(["--output-dir", str(output_dir)])
    return argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TBME visual outputs.",
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="output_set")

    # Summary : Write summary figures from existing summary CSV files.
    summary = subparsers.add_parser(
        "summary",
        help="Write per-suite summary and trajectory figures.",
    )
    summary.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME group names.",
    )
    summary.add_argument("--figure-formats", type=str, default=".pdf")
    summary.add_argument("--trajectory-max-seeds", type=int, default=50)
    summary.add_argument("--density-bins", type=int, default=96)
    summary.add_argument("--results-dir", type=str, default=None)

    # Overview : Write group-level overview tables and figures.
    overview = subparsers.add_parser(
        "overview",
        help="Write group-level overview tables and figures into each group session.",
    )
    overview.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME group names.",
    )
    overview.add_argument("--results-dir", type=str, default=None)

    experiment = subparsers.add_parser(
        "experiment",
        help="Generate experiment-level figures into suite result folders.",
    )
    experiment.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME experiment group names.",
    )
    experiment.add_argument("--max-seeds", type=int, default=100)
    experiment.add_argument("--results-dir", type=str, default=None)

    assets = subparsers.add_parser(
        "assets",
        help="Prepare TBME manuscript asset assembly outputs.",
    )
    assets.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help="Comma-separated TBME group names.",
    )
    assets.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for assembled manuscript assets.",
    )
    assets.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="TBME results root. Defaults to results/tbme.",
    )

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
    all_parser.add_argument("--results-dir", type=str, default=None)
    all_parser.add_argument(
        "--assets-output-dir",
        type=str,
        default=None,
        help="Directory for assembled manuscript assets.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_set is None:
        parser.print_help()
        return 0
    if getattr(args, "results_dir", None) is not None:
        tbme_figures._set_tbme_results_dir(Path(args.results_dir))
    if args.output_set == "summary":
        return int(tbme_figures.summary_main(_summary_args(args)))
    if args.output_set == "overview":
        return int(tbme_figures.group_overview_main(_overview_args(args)))
    if args.output_set == "experiment":
        return int(tbme_figures.experiment_main(_experiment_args(args)))
    if args.output_set == "assets":
        return int(tbme_figures.assets_main(_assets_args(args)))
    if args.output_set == "all":
        args.trajectory_max_seeds = (
            args.trajectory_max_seeds if args.trajectory_max_seeds is not None else args.max_seeds
        )
        code = int(tbme_figures.summary_main(_summary_args(args)))
        if code != 0:
            return code
        code = int(tbme_figures.group_overview_main(_overview_args(args)))
        if code != 0:
            return code
        code = int(tbme_figures.experiment_main(_experiment_args(args)))
        if code != 0:
            return code
        code = int(tbme_figures.assets_main(_assets_args(args)))
        if code != 0:
            return code
        return 0
    raise ValueError(f"Unknown output set: {args.output_set}")


if __name__ == "__main__":
    raise SystemExit(main())
