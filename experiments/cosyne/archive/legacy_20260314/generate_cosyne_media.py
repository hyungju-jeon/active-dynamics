#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_extra_figure(summary_dir: Path) -> Path | None:
    csv_path = summary_dir / "trajectory_r2_over_steps.csv"
    if not csv_path.exists():
        return None

    df = pd.read_csv(csv_path).dropna(subset=["cpu_time_sec_mean", "trajectory_r2_mean"])
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for (model_tag, exp_id), sub in df.groupby(["model_tag", "exp_id"]):
        ax.plot(
            sub["cpu_time_sec_mean"],
            sub["trajectory_r2_mean"],
            label=f"{model_tag}:{exp_id}",
        )
    ax.set_xlabel("CPU Time (sec)")
    ax.set_ylabel("Trajectory R2")
    ax.set_title("Trajectory R2 over CPU time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    out_path = summary_dir / "figures" / "trajectory_r2_over_cpu_time.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper that reuses the shared COSYNE session renderers."
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--summary-dir", required=True)
    parser.add_argument("--model-tag", default="updated")
    parser.add_argument("--exp-id", default="active_short")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", default="repeat_01")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--grid-lim", type=float, default=10.0)
    parser.add_argument("--ig-grid", type=int, default=121)
    parser.add_argument("--ig-dt", type=float, default=0.01)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from generate_session_behavior_video import make_acq_action_video, make_info_maps_video

    base_dir = Path(args.base_dir)
    summary_dir = Path(args.summary_dir)
    run_dir = (
        base_dir / "tracks" / args.model_tag / args.exp_id / f"seed_{args.seed}" / args.repeat
    )
    media_dir = summary_dir / "media"
    is_passive = str(args.exp_id).strip().lower() in {"random", "no_policy"}

    info_path = None
    if not is_passive:
        info_path = make_info_maps_video(
            run_dir=run_dir,
            output_path=media_dir / "info_gain_proxy_over_time.mp4",
            stride=max(1, int(args.stride)),
            fps=max(1, int(args.fps)),
            grid_lim=float(args.grid_lim),
            ig_grid=max(25, int(args.ig_grid)),
            ig_dt=float(args.ig_dt),
        )
    behavior_path = make_acq_action_video(
        run_dir=run_dir,
        output_path=media_dir / "session_behavior_overlay.mp4",
        stride=max(1, int(args.stride)),
        fps=max(1, int(args.fps)),
        grid_lim=float(args.grid_lim),
    )
    extra_path = make_extra_figure(summary_dir)

    if info_path is not None:
        print(info_path)
    print(behavior_path)
    if extra_path is not None:
        print(extra_path)


if __name__ == "__main__":
    main()
