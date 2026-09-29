"""Open-loop checks of the fitted spiking-network surrogate.

The coefficients stay fixed at reference/m2_fit.json. Predictions use only
the recorded initial state and inputs, with no spike-count filtering.
Input arrays are the 12 evaluation trajectories in data/test_trajectories.npz;
their seed order is checked against the recorded input-generation protocol.

Regenerate:
    .venv/bin/python -m experiments.tnsre.figures.spiking_validation \
        --experiment-dir results/tnsre/20260924_snn_sessions_m2 \
        --out-dir results/tnsre/20260925_snn_surrogate_validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from actdyn.utils.figure_io import load_plotting
from actdyn.utils.validation import rollout_r2_on_trajectories
from experiments.tnsre.eval_spiking_sessions import (
    PROTOCOL,
    learner_transition,
    test_input_program,
)

from .assets import _apply_asset_style, save_figure
from .theme import style_experiment_axis

WIDTH_IN = 516.0 / 72.27
HEIGHT_IN = 3.45
STEM = "appendix_spiking_validation"
POOL_COLORS = ("#3E6FB0", "#C9562C")
CONDITIONS = ("500_ms_windows", "first_500_ms", "full_2_s")
DT = 0.05  # In units of the 100-ms NMDA time constant.
BIN_SECONDS = 0.005


def open_loop(theta: np.ndarray, starts: np.ndarray, inputs: np.ndarray) -> np.ndarray:
    """Roll out (..., 2) states under (..., T, 2) inputs, including the start."""
    transition = learner_transition(
        "wong_wang_inside_gain", theta, theta, 5, DT, torch.float32
    )
    x = torch.as_tensor(starts, dtype=torch.float32)
    drive = torch.as_tensor(inputs, dtype=torch.float32)
    path = [x.numpy().copy()]
    with torch.no_grad():
        for k in range(inputs.shape[-2]):
            x = transition(x, drive[..., k, :])
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"Non-finite prediction at bin {k + 1}")
            path.append(x.numpy().copy())
    return np.stack(path, axis=-2)


def trajectory_scores(predicted: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-network-seed and pooled R2; each denominator uses its own target mean."""
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    axes = tuple(range(1, target.ndim))
    sse = np.sum((predicted - target) ** 2, axis=axes)
    sst = np.sum((target - target.mean(axis=axes, keepdims=True)) ** 2, axis=axes)
    if np.any(sst <= 0):
        raise ValueError("R2 is undefined for a constant target trajectory.")
    pooled_sst = np.sum((target - target.mean()) ** 2)
    return 1 - sse / sst, float(1 - sse.sum() / pooled_sst)


def evaluate(experiment_dir: Path) -> dict[str, Any]:
    """Read saved trajectories and compare three fixed-parameter rollout protocols."""
    data_path = experiment_dir / "data" / "test_trajectories.npz"
    fit_path = experiment_dir / "reference" / "m2_fit.json"
    with np.load(data_path) as saved:
        states, inputs = saved["states"], saved["inputs"]
    record = json.loads(fit_path.read_text())
    theta = np.asarray(record["learner_parameters"]["values"], dtype=np.float64)
    seeds = np.asarray(PROTOCOL["test_seeds"], dtype=int)
    if states.shape != (12, 401, 2) or inputs.shape != (12, 400, 2):
        raise ValueError(f"Unexpected evaluation shapes: {states.shape}, {inputs.shape}")
    np.testing.assert_array_equal(inputs, np.stack([test_input_program(s) for s in seeds]))
    if not np.isfinite(states).all() or not np.isfinite(inputs).all():
        raise ValueError("Non-finite recorded data.")

    # Established metric: 500-ms predictions initialized at recorded states
    # every 100 ms. Each window is propagated independently without correction.
    window_starts = np.arange(0, inputs.shape[1] - 100 + 1, 20)
    initial_states = np.stack([states[:, s] for s in window_starts], axis=1)
    window_inputs = np.stack([inputs[:, s:s + 100] for s in window_starts], axis=1)
    window_target = np.stack([states[:, s:s + 101] for s in window_starts], axis=1)
    window_prediction = open_loop(theta, initial_states, window_inputs)
    full_prediction = open_loop(theta, states[:, 0], inputs)
    per_seed, pooled = [], []
    for predicted, target in (
        (window_prediction[:, :, 1:], window_target[:, :, 1:]),
        (full_prediction[:, 1:101], states[:, 1:101]),
        (full_prediction[:, 1:], states[:, 1:]),
    ):
        score, aggregate = trajectory_scores(predicted, target)
        per_seed.append(score)
        pooled.append(aggregate)
    per_seed, pooled = np.stack(per_seed, axis=1), np.asarray(pooled)

    # Independent comparison to the manuscript's established scoring function.
    canonical = rollout_r2_on_trajectories(
        torch.as_tensor(theta[None], dtype=torch.float32),
        dynamics_type="wong_wang_inside_gain", full_params=theta,
        min_embedding_dim=5, states=states, inputs=inputs, dt=DT,
        horizon=100, stride=20,
    )[0]
    np.testing.assert_allclose(pooled[0], canonical, rtol=0, atol=2e-7)
    np.testing.assert_allclose(pooled[0], record["r2_test_piecewise"], rtol=0, atol=2e-7)
    np.testing.assert_array_equal(full_prediction[:, 0], states[:, 0])
    np.testing.assert_array_equal(window_prediction[:, :, 0], initial_states)
    # Different batch widths can change float32 vectorized rounding.
    np.testing.assert_allclose(
        window_prediction[:, 0], full_prediction[:, :101], rtol=0, atol=1e-6
    )

    # Show the lower median rank and the minimum, never hand-picked trajectories.
    order = np.lexsort((seeds, per_seed[:, 2]))
    example_indices = np.asarray([order[(len(order) - 1) // 2], order[0]])
    return dict(
        states=states, inputs=inputs, theta=theta, seeds=seeds,
        window_starts=window_starts, window_target=window_target,
        window_prediction=window_prediction, full_prediction=full_prediction,
        per_seed_r2=per_seed, pooled_r2=pooled, example_indices=example_indices,
        source_paths=[data_path, fit_path], canonical_r2=float(canonical),
    )


def build_figure(data: dict[str, Any], plt: Any, *, grayscale: bool = False) -> Any:
    """Draw two complete example trajectories and all 12 paired seed scores."""
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    colors = ("#303030", "#686868") if grayscale else POOL_COLORS
    time = np.arange(data["states"].shape[1]) * BIN_SECONDS
    y_min = min(float(data["states"].min()), float(data["full_prediction"].min())) - 0.1
    y_max = max(float(data["states"].max()), float(data["full_prediction"].max())) + 0.1
    for col, (left, idx, title) in enumerate(zip(
        (0.075, 0.400), data["example_indices"],
        ("Median-ranked trajectory", "Lowest-scoring trajectory"),
    )):
        fig.text(left - 0.034, 0.957, "AB"[col], fontsize=10, weight="bold")
        fig.text(left, 0.960, title, fontsize=8)
        score = data["per_seed_r2"][idx, 2]
        fig.text(left, 0.893, f"Seed {data['seeds'][idx]}; 2-s $R^2={score:.3f}$", fontsize=6)
        drive_ax = fig.add_axes((left, 0.16, 0.245, 0.12))
        drive_ax.set_xlim(0, 2)
        drive_ax.set_xticks([0, 0.5, 1, 1.5, 2])
        drive_ax.set_xlabel("Time (s)")
        drive_ax.set_ylabel("Input (pA)")
        drive_ax.set_ylim(-24, 24)
        drive_ax.set_yticks([-20, 0, 20])
        style_experiment_axis(drive_ax)
        for pool, bottom in enumerate((0.65, 0.39)):
            ax = fig.add_axes((left, bottom, 0.245, 0.20), sharex=drive_ax)
            ax.plot(time, data["states"][idx, :, pool], color="#909090", lw=1.1)
            ax.plot(time, data["full_prediction"][idx, :, pool],
                    color=colors[pool], lw=1.05, ls="--")
            ax.set_ylim(y_min, y_max)
            ax.set_yticks([-2, -1, 0, 1])
            ax.set_ylabel(rf"State $z_{pool + 1}$")
            ax.tick_params(axis="x", labelbottom=False)
            style_experiment_axis(ax)
            # Steps extend through the final observed bin; no interpolation or smoothing.
            drive = 20 * data["inputs"][idx, :, pool]
            drive_ax.step(time, np.r_[drive, drive[-1]], where="post",
                          color=colors[pool], lw=0.75, ls=("-", ":")[pool],
                          marker=("s", "o")[pool], markevery=40, ms=2)

    fig.text(0.707, 0.957, "C", fontsize=10, weight="bold")
    fig.text(0.747, 0.960, "All 12 test trajectories", fontsize=8)
    fig.text(0.747, 0.893, "Pooled: " + " / ".join(f"{s:.3f}" for s in data["pooled_r2"]),
             fontsize=6)
    score_ax = fig.add_axes((0.750, 0.20, 0.235, 0.65))
    positions = np.arange(3)
    for row in data["per_seed_r2"]:
        score_ax.plot(positions, row, color="#AAAAAA", lw=0.55, marker="o",
                      ms=2.4, markerfacecolor="white", markeredgewidth=0.5, alpha=0.8)
    score_ax.plot(positions, data["pooled_r2"], color="#222222", lw=1.1,
                  marker="D", ms=4, zorder=5)
    score_ax.axhline(0, color="#777777", ls=":", lw=0.65, zorder=0)
    score_ax.set_xlim(-0.22, 2.22)
    score_ax.set_ylim(min(-0.15, float(data["per_seed_r2"].min()) - 0.25), 1.15)
    score_ax.set_xticks(positions, ["500 ms\nall starts", "500 ms\nonset", "2 s\nonset"])
    score_ax.set_xlabel("Prediction protocol")
    score_ax.set_ylabel(r"Prediction $R^2$")
    style_experiment_axis(score_ax)

    handles = [
        Line2D([], [], color="#909090", lw=1.1, label="Network state"),
        Line2D([], [], color="#333333", lw=1.05, ls="--", label="Reduced model"),
        Line2D([], [], color=colors[0], lw=0.75, marker="s", ms=2, label="Input 1"),
        Line2D([], [], color=colors[1], lw=0.75, ls=":", marker="o", ms=2, label="Input 2"),
        Line2D([], [], color="#AAAAAA", lw=0.55, marker="o", mfc="white",
               ms=2.4, label="One test trajectory"),
        Line2D([], [], color="#222222", lw=1.1, marker="D", ms=4, label="Pooled score"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.51, 0.015),
               ncol=6, frameon=False, fontsize=6, handlelength=1.8, columnspacing=1.1)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    data = evaluate(args.experiment_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plt = load_plotting(args.out_dir, apply_style=_apply_asset_style)
    if plt is None:
        raise RuntimeError("Matplotlib is unavailable.")
    fig = build_figure(data, plt)
    fig.savefig(args.out_dir / f"{STEM}.png", dpi=200, bbox_inches=None)
    save_figure(fig, args.out_dir / f"{STEM}.pdf", plt_module=plt)
    gray = build_figure(data, plt, grayscale=True)
    gray.savefig(args.out_dir / f"{STEM}_grayscale.png", dpi=200, bbox_inches=None)
    plt.close(gray)
    arrays = {k: v for k, v in data.items() if isinstance(v, np.ndarray)}
    np.savez_compressed(args.out_dir / "rollouts.npz", **arrays)
    with (args.out_dir / "scores.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["network_seed", *CONDITIONS])
        writer.writerows([int(seed), *row] for seed, row in zip(data["seeds"], data["per_seed_r2"]))
        writer.writerow(["pooled", *data["pooled_r2"]])
    metadata = dict(
        sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in data["source_paths"]],
        generator=str(Path(__file__)),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        normalized_dt=DT, bin_seconds=BIN_SECONDS,
        state_definition="z_i = 4 (mean_NMDA_gate_i - 1/2)",
        parameters="Fixed privileged reference; no refitting or online learning.",
        filtering="None. Each rollout uses its recorded initial state and inputs only.",
        scores=dict(zip(CONDITIONS, data["pooled_r2"].tolist())),
        canonical_500_ms_r2=data["canonical_r2"],
        aggregation="1-SSE/SST; own target mean per seed, global target mean when pooled.",
        examples=dict(median_rank=6, total=12, indices=data["example_indices"].tolist(),
                      seeds=data["seeds"][data["example_indices"]].tolist(),
                      ranking="Ascending full-2-s R2; ties broken by network seed."),
        sampling="12 network seeds; no smoothing, intervals, exclusions, or score clipping.",
        window_protocol="100-bin windows, starts 0:20:300; overlapping windows are not independent samples.",
        selection_scope="Separate from coefficient fitting; this evaluation set was used in the pilot model comparison.",
        font_rule="Helvetica/Nimbus Sans; panel letters 10 pt bold, labels 8 pt, ticks/legends 6 pt.",
        size_inches=[WIDTH_IN, HEIGHT_IN],
    )
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"figure": str(args.out_dir / f"{STEM}.pdf"),
                      "pooled_r2": metadata["scores"], "example_seeds": metadata["examples"]["seeds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
