#!/usr/bin/env python3
from __future__ import annotations

import argparse
from experiments.tbme import visualize


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TBME visual outputs.")
    subparsers = parser.add_subparsers(dest="output_set")

    summary = subparsers.add_parser(
        "summary",
        help="Regenerate per-suite summary figures from existing summary CSV files.",
    )
    summary.add_argument(
        "--groups",
        type=str,
        default="exp1_main,exp1_schedule,exp1_hard,exp2",
        help="Comma-separated TBME group names.",
    )
    summary.add_argument("--figure-formats", type=str, default=".pdf")

    subparsers.add_parser(
        "assets",
        help="Export current manuscript tables and copied summary assets.",
    )

    trajectory = subparsers.add_parser(
        "trajectory",
        help="Generate trajectory overlay and density summary figures.",
    )
    trajectory.add_argument(
        "--groups",
        type=str,
        default="exp1_main,exp1_schedule,exp1_hard,exp2",
        help="Comma-separated TBME group names.",
    )
    trajectory.add_argument("--max-seeds", type=int, default=50)
    trajectory.add_argument("--density-bins", type=int, default=96)

    requested = subparsers.add_parser(
        "requested",
        help="Generate requested follow-up experiment figures.",
    )
    requested.add_argument(
        "--additional-session",
        type=str,
        default=None,
        help="Session directory containing completed additional follow-up suites.",
    )

    additional = subparsers.add_parser(
        "additional",
        help="Generate additional manuscript figures.",
    )
    additional.add_argument("--max-seeds", type=int, default=100)

    all_parser = subparsers.add_parser("all", help="Generate all TBME visual outputs.")
    all_parser.add_argument(
        "--groups",
        type=str,
        default="exp1_main,exp1_schedule,exp1_hard,exp2",
        help="Comma-separated TBME group names for summary and trajectory figures.",
    )
    all_parser.add_argument("--figure-formats", type=str, default=".pdf")
    all_parser.add_argument("--max-seeds", type=int, default=100)
    all_parser.add_argument("--trajectory-max-seeds", type=int, default=None)
    all_parser.add_argument("--density-bins", type=int, default=96)
    all_parser.add_argument(
        "--additional-session",
        type=str,
        default=None,
        help="Session directory containing completed additional follow-up suites.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.output_set is None:
        parser.print_help()
        return 0
    if args.output_set == "summary":
        return visualize.run_summary_figure_regeneration(
            ["--groups", str(args.groups), "--figure-formats", str(args.figure_formats)]
        )
    if args.output_set == "assets":
        return visualize.run_current_asset_export([])
    if args.output_set == "trajectory":
        return visualize.run_trajectory_summary_figures(
            [
                "--groups",
                str(args.groups),
                "--max-seeds",
                str(args.max_seeds),
                "--density-bins",
                str(args.density_bins),
            ]
        )
    if args.output_set == "requested":
        requested_args = (
            []
            if args.additional_session is None
            else ["--additional-session", str(args.additional_session)]
        )
        return visualize.run_requested_experiment_figures(requested_args)
    if args.output_set == "additional":
        return visualize.run_additional_manuscript_figures(["--max-seeds", str(args.max_seeds)])
    if args.output_set == "all":
        trajectory_max_seeds = (
            args.trajectory_max_seeds if args.trajectory_max_seeds is not None else args.max_seeds
        )
        requested_args = (
            []
            if args.additional_session is None
            else ["--additional-session", str(args.additional_session)]
        )
        for runner, runner_args in (
            (
                visualize.run_summary_figure_regeneration,
                ["--groups", str(args.groups), "--figure-formats", str(args.figure_formats)],
            ),
            (visualize.run_current_asset_export, []),
            (
                visualize.run_trajectory_summary_figures,
                [
                    "--groups",
                    str(args.groups),
                    "--max-seeds",
                    str(trajectory_max_seeds),
                    "--density-bins",
                    str(args.density_bins),
                ],
            ),
            (visualize.run_requested_experiment_figures, requested_args),
            (
                visualize.run_additional_manuscript_figures,
                ["--max-seeds", str(args.max_seeds)],
            ),
        ):
            code = int(runner(runner_args))
            if code != 0:
                return code
        return 0
    raise ValueError(f"Unknown output set: {args.output_set}")


if __name__ == "__main__":
    raise SystemExit(main())
