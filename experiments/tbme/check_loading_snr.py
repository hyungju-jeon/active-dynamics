#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections.abc import Sequence

from actdyn.utils.experiment_runtime import compute_loglinear_loading_fisher_snr_db
from experiments.tbme.run_tbme_experiments import configure_tbme_catalogs


def main(argv: Sequence[str] | None = None) -> int:
    """Check achieved Fisher SNR against each TBME environment target.

    The check uses the same log-linear observation initialization path as the
    runtime experiments. For each TBME environment preset with
    ``loading_target_snr_db``, it recomputes the Fisher SNR from the final
    loading matrix and bias.
    """
    parser = argparse.ArgumentParser(
        description="Check TBME log-linear loading SNR targets.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--loading-seed",
        type=int,
        default=0,
        help="Seed used for the shared observation loading initialization.",
    )
    parser.add_argument(
        "--snr-seed",
        type=int,
        default=0,
        help="Seed used for target-SNR calibration trajectories.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help="Number of zero-action trajectories used for SNR calibration.",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=500,
        help="Number of latent states per SNR calibration trajectory.",
    )
    parser.add_argument(
        "--abs-tol-db",
        type=float,
        default=1.0,
        help="Allowed absolute SNR error in dB.",
    )
    args = parser.parse_args(None if argv is None else list(argv))

    bundle = configure_tbme_catalogs()
    rows: list[tuple[str, float, float, float, bool]] = []
    for preset_id, env_preset in sorted(bundle.environment_presets.items()):
        if not str(preset_id).startswith("tbme_"):
            continue
        target = getattr(env_preset, "loading_target_snr_db", None)
        if target is None:
            continue
        with contextlib.redirect_stdout(sys.stderr):
            achieved = compute_loglinear_loading_fisher_snr_db(
                env_preset,
                seed=int(args.loading_seed),
                snr_seed=int(args.snr_seed),
                num_trajectories=int(args.num_trajectories),
                trajectory_length=int(args.trajectory_length),
            )
        diff = float(achieved) - float(target)
        rows.append(
            (
                str(preset_id),
                float(target),
                float(achieved),
                diff,
                math.isfinite(diff) and abs(diff) <= float(args.abs_tol_db),
            )
        )

    if not rows:
        print("No TBME environments define loading_target_snr_db.")
        return 0

    print(
        f"{'environment':58s} {'target_db':>10s} {'achieved_db':>12s} "
        f"{'diff_db':>10s} status"
    )
    for preset_id, target, achieved, diff, ok in rows:
        status = "ok" if ok else "FAIL"
        print(
            f"{preset_id:58s} {target:10.3f} {achieved:12.4f} "
            f"{diff:10.4f} {status}"
        )

    failed = [preset_id for preset_id, *_rest, ok in rows if not ok]
    print(
        f"\n{len(rows) - len(failed)}/{len(rows)} environments met "
        f"abs_tol_db={float(args.abs_tol_db):.3f}."
    )
    if failed:
        print("Failed environments: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
