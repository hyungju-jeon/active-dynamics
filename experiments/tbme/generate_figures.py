#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
from pathlib import Path
import sys


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from experiments.tbme import tbme_figures
from experiments.tbme import tbme_figures_experiment
from experiments.tbme.run_tbme_experiments import SHARED_TBME_GROUP_MODULES


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
DIAGNOSTIC_ENV_IDS = (
    "tbme_duffing",
    "tbme_damped_pendulum",
    "tbme_gated_duffing",
    "tbme_gated_duffing_asymmetric",
    "tbme_gated_duffing_observation_bottleneck_strong",
)


def _add_groups_arg(
    parser: argparse.ArgumentParser,
    *,
    help_text: str = "Comma-separated TBME group names.",
) -> None:
    parser.add_argument(
        "--groups",
        type=str,
        default=DEFAULT_GROUPS,
        help=help_text,
    )


def _add_figure_formats_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--figure-formats", type=str, default=".pdf")


def _add_results_dir_arg(
    parser: argparse.ArgumentParser,
    *,
    help_text: str | None = None,
) -> None:
    parser.add_argument("--results-dir", type=str, default=None, help=help_text)


def _add_max_seeds_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-seeds", type=int, default=100)


def _add_trajectory_args(
    parser: argparse.ArgumentParser,
    *,
    max_seeds_default: int | None,
) -> None:
    parser.add_argument("--trajectory-max-seeds", type=int, default=max_seeds_default)
    parser.add_argument("--density-bins", type=int, default=96)


def _config_items(raw: str | Sequence[str] | None) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return _csv_items(raw)
    return [str(item) for item in raw]


def _selection_from_groups(group_csv: str) -> str:
    group_ids = _csv_items(group_csv)
    group_modules = dict(SHARED_TBME_GROUP_MODULES)
    unknown = sorted(set(group_ids) - set(group_modules))
    if unknown:
        raise ValueError(f"Unknown group(s): {', '.join(unknown)}")

    items: list[str] = []
    seen: set[tuple[str, str]] = set()
    for group_id in group_ids:
        for module_ref in group_modules[group_id]:
            module = importlib.import_module(module_ref)
            suites = getattr(module, "EXPERIMENT_SUITES")
            exp_ids = _config_items(getattr(module, "DEFAULT_EXP_IDS", None)) or list(suites)
            for exp_id in exp_ids:
                spec = suites[exp_id]
                env_preset_id = str(spec["env_preset_id"])
                for policy_id in _config_items(spec.get("model_ids", spec.get("policy_ids"))):
                    key = (env_preset_id, policy_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(f"{env_preset_id}:{policy_id}")
    return ",".join(items)


def _summary_args(args: argparse.Namespace) -> list[str]:
    return [
        "--selection",
        _selection_from_groups(str(args.groups)),
        "--figure-formats",
        str(args.figure_formats),
        "--trajectory-max-seeds",
        str(args.trajectory_max_seeds),
        "--density-bins",
        str(args.density_bins),
    ]


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
    unknown = sorted(set(plot_ids) - set(tbme_figures_experiment.EXPERIMENT_PLOTS))
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


def _diagnostics_args(args: argparse.Namespace) -> list[str]:
    argv = [
        "--env-ids",
        ",".join(DIAGNOSTIC_ENV_IDS),
        "--figure-formats",
        str(args.figure_formats),
        "--steps",
        str(args.steps),
        "--trajectories",
        str(args.trajectories),
        "--seed",
        str(args.seed),
        "--grid",
        str(args.grid),
        "--snr-trajectories",
        str(args.snr_trajectories),
        "--snr-trajectory-length",
        str(args.snr_trajectory_length),
    ]
    if args.output_dir is not None:
        argv.extend(["--output-dir", str(args.output_dir)])
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
    _add_groups_arg(summary)
    _add_figure_formats_arg(summary)
    _add_trajectory_args(summary, max_seeds_default=50)
    _add_results_dir_arg(summary)

    experiment = subparsers.add_parser(
        "experiment",
        help="Generate experiment-level figures into suite result folders.",
    )
    _add_groups_arg(experiment, help_text="Comma-separated TBME experiment group names.")
    _add_max_seeds_arg(experiment)
    _add_results_dir_arg(experiment)

    assets = subparsers.add_parser(
        "assets",
        help="Prepare TBME manuscript asset assembly outputs.",
    )
    _add_groups_arg(assets)
    assets.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for assembled manuscript assets.",
    )
    _add_results_dir_arg(assets, help_text="TBME results root. Defaults to results/tbme.")

    diagnostics = subparsers.add_parser(
        "diagnostics",
        help="Generate dynamics and observation diagnostics from TBME env catalogs.",
    )
    _add_results_dir_arg(diagnostics)
    diagnostics.add_argument("--output-dir", type=str, default=None)
    diagnostics.add_argument("--figure-formats", type=str, default=".pdf")
    diagnostics.add_argument("--steps", type=int, default=500)
    diagnostics.add_argument("--trajectories", type=int, default=3)
    diagnostics.add_argument("--seed", type=int, default=0)
    diagnostics.add_argument("--grid", type=int, default=51)
    diagnostics.add_argument("--snr-trajectories", type=int, default=100)
    diagnostics.add_argument("--snr-trajectory-length", type=int, default=200)

    all_parser = subparsers.add_parser("all", help="Generate all TBME visual outputs.")
    _add_groups_arg(all_parser)
    _add_figure_formats_arg(all_parser)
    _add_max_seeds_arg(all_parser)
    _add_trajectory_args(all_parser, max_seeds_default=None)
    _add_results_dir_arg(all_parser)
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
    if args.output_set == "experiment":
        return int(tbme_figures.experiment_main(_experiment_args(args)))
    if args.output_set == "assets":
        return int(tbme_figures.assets_main(_assets_args(args)))
    if args.output_set == "diagnostics":
        return int(tbme_figures.diagnostics_main(_diagnostics_args(args)))
    if args.output_set == "all":
        args.trajectory_max_seeds = (
            args.trajectory_max_seeds if args.trajectory_max_seeds is not None else args.max_seeds
        )
        code = int(tbme_figures.summary_main(_summary_args(args)))
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
