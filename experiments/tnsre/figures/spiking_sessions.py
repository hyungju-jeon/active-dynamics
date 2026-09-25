"""Figures of the spiking-network decision experiment (Wang 2002 network, M2 learner).

Reads the outputs of ``experiments.tnsre.eval_spiking_sessions`` and
``experiments.tnsre.diag_control_gap`` under one experiment folder and draws,
in the manuscript asset style:

* ``identification`` (main text): (A) circuit above (B) session protocol, (C)
  sessions 1 and 20 of one PALDI run (spike raster of the 80 observed neurons,
  network latent and filtered state, input), (D) held-out rollout R2 after each
  identification session (median, IQR over 20 training seeds; dotted = reduced
  model fitted to network data).
* ``control`` (main text, warm-start controller): (A) task timelines, (B, C)
  force and overturn success after 20 sessions versus the energy budget
  (mean +/- SEM over seeds) with the model-free pushes and the fitted model on
  the same task sessions, (D, E) success along identification at one budget per
  task, (F) one overturn session: target-pool lead and input under PALDI's
  model, the fitted model, the full push, and no input.
* ``diagnosis`` (appendix, fresh controller): (A) held-out R2 on network
  trajectories of the control regime by push direction, (B) overturn success
  with one parameter exchanged between PALDI's estimate and the fit, or with the
  planner and the filter on different models, (C) held-out R2 versus overturn
  success per training seed, (D) learned cross-inhibition and gain scale against
  the fit and batch refits on PALDI's session data.

Usage:
    python -m experiments.tnsre.figures.spiking_sessions --experiment-dir results/tnsre/20260924_snn_sessions_m2 \\
        --figure all
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from actdyn.utils.figure_io import load_plotting

from .assets import (
    _ASSET_LABEL_SIZE,
    _ASSET_PANEL_LABEL_SIZE,
    _ASSET_TICK_SIZE,
    _apply_asset_style,
    _asset_baseline_policy_color,
    _asset_policy_label,
    save_figure,
)
from .theme import NEUTRAL_LIGHT, STROKE_COLOR, style_experiment_axis

POLICIES = ("adaptive", "active_myopic", "flex_rollback", "rhc", "prbs", "random")
ACTIVE = ("adaptive", "active_myopic", "rhc")
EXP_ID = "wong_wang_snn_sessions_m2"
FINAL_SESSIONS = 20
POOL_COLORS = ("#3E6FB0", "#C9562C")
# Area shading shared by the spiking figures: identification input (gray), control input in the
# tasks (green), and the decision with its post-decision hold (purple). Evidence is NEUTRAL_LIGHT.
PROBE_SHADE, PROBE_TEXT = "#E8E5E0", "#6E6861"
CONTROL_SHADE = "#DCEBDF"
DECISION_SHADE, DECISION_TEXT = "#DDD2EC", "#745197"
DECISION_ALPHA = 0.45
FIGURE_WIDTH = 516.0 / 72.27  # IEEE text width in inches
REFERENCE_STYLE = {
    "reduced_fit": dict(color="#4A4A4A", linestyle=":", marker="s", label="model fitted to network data"),
    "spread": dict(color="#8C8C8C", linestyle="--", marker="^", label="even push"),
    "front": dict(color="#8B6B4A", linestyle="-.", marker="v", label="full push from onset"),
}
DECISION_MARKER = dict(marker="o", ms=3.0, markeredgecolor="white", markeredgewidth=0.4)
T_SHOWN_MS = 2500.0  # example session axis; both unsuccessful traces stay below the threshold after it
TASK_TITLES = {"force": "Force", "overturn": "Overturn"}
CURVE_BUDGETS = {"force": 1.0, "overturn": 12.0}


# ------------------------------------------------------------------ data helpers
def _read(path: Path) -> list[dict[str, str]]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _summary_lookup(path: Path) -> dict[tuple[str, int, str, float], tuple[float, float]]:
    return {(r["policy"], int(r["checkpoint"]), r["task"], float(r["budget"])):
            (float(r["success"]), float(r["sem_over_seeds"])) for r in _read(path)}


def _softplus(x: float) -> float:
    return float(np.log1p(np.exp(x)))


def _panel_label(ax: Any, letter: str, dx: float = -7.2) -> None:
    ax.annotate(letter, (0, 1), xycoords="axes fraction", xytext=(dx, 3.0), textcoords="offset points",
                ha="left", va="bottom", fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold")


def _r2_by_session(rows: list[dict[str, str]]) -> dict[tuple[str, int], list[float]]:
    out: dict[tuple[str, int], list[float]] = {}
    for r in rows:
        out.setdefault((r["policy"], int(r["sessions"])), []).append(float(r["r2"]))
    return out


def _per_seed_success(task_dir: Path, task: str, budget: float) -> dict[tuple[str, int, int], float]:
    """Mean success per (policy, training seed, checkpoint) over the valid task sessions."""
    acc: dict[tuple[str, int, int], list[int]] = {}
    for path in sorted(task_dir.glob("worker_*.csv")):
        for r in _read(path):
            if r["valid"] == "1" and r["task"] == task and float(r["budget"]) == budget and r["controller"] == "icem":
                acc.setdefault((r["policy"], int(r["seed"]), int(r["checkpoint"])), []).append(int(r["success"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ------------------------------------------------------------------ schematics
def _draw_circuit(ax: Any) -> None:
    """Two selective pools with recurrent excitation, shared inhibition, inputs, and recordings."""
    from matplotlib.patches import Circle, FancyArrowPatch

    ax.set_xlim(0, 12)
    ax.set_ylim(0.95, 6.25)
    ax.set_aspect("equal", anchor="N")
    ax.axis("off")
    ax.set_xticks([])
    ax.set_yticks([])
    # A shallow triangle uses the column width; input and recording leads enter
    # from the sides, keeping the circuit compact without scaling its text.
    pools = {1: (3.1, 4.3), 2: (8.9, 4.3)}
    radius = 1.1
    pool_patches = {}
    for i, (x, y) in pools.items():
        pool = Circle((x, y), radius, facecolor=POOL_COLORS[i - 1], alpha=0.18,
                      edgecolor=POOL_COLORS[i - 1], lw=1.0)
        ax.add_patch(pool)
        pool_patches[i] = pool
        ax.text(x, y, f"pool {i}", ha="center", va="center", fontsize=_ASSET_TICK_SIZE, color=STROKE_COLOR)
        side = -1 if i == 1 else 1
        loop = FancyArrowPatch((x - 0.7, y + 0.85), (x + 0.7, y + 0.85),
                               connectionstyle="arc3,rad=-0.9", arrowstyle="-|>", mutation_scale=6,
                               lw=0.8, color=POOL_COLORS[i - 1])
        ax.add_patch(loop)
        ax.annotate("", xy=(x + side * 1.0, y + 0.55), xytext=(x + side * 2.2, y + 1.3),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=STROKE_COLOR, mutation_scale=7))
        ax.text(x + side * 2.5, y + 1.45, f"$u_{i}$", ha="center", va="center", fontsize=_ASSET_LABEL_SIZE)
        ax.plot([x + side * 0.95, x + side * 1.9], [y - 0.65, y - 0.7], color=STROKE_COLOR, lw=0.7)
        ax.text(x + side * 2.1, y - 1.3, "40 rec.", ha="center", va="center", fontsize=_ASSET_TICK_SIZE,
                color=STROKE_COLOR)
    inhibitory_center = (6.0, 3.1)
    inhibitory = Circle(inhibitory_center, 0.7, facecolor=NEUTRAL_LIGHT, alpha=0.5,
                        edgecolor=STROKE_COLOR, lw=0.8)
    ax.add_patch(inhibitory)
    ax.text(*inhibitory_center, "I", ha="center", va="center", fontsize=_ASSET_TICK_SIZE, color=STROKE_COLOR)
    for i, center in pools.items():
        ax.add_patch(FancyArrowPatch(center, inhibitory_center, patchA=pool_patches[i], patchB=inhibitory,
                                    connectionstyle="arc3,rad=0.16", arrowstyle="-|>", mutation_scale=6,
                                    shrinkA=1.5, shrinkB=1.5, lw=0.7, color=STROKE_COLOR))
        ax.add_patch(FancyArrowPatch(inhibitory_center, center, patchA=inhibitory, patchB=pool_patches[i],
                                    connectionstyle="arc3,rad=0.16", arrowstyle="-[", mutation_scale=3,
                                    shrinkA=1.5, shrinkB=1.5, lw=0.7, color=STROKE_COLOR))
    ax.text(6.0, 1.25, "Wang (2002): 240 E/pool, 400 I", ha="center", va="bottom", fontsize=_ASSET_TICK_SIZE,
            color=STROKE_COLOR)


def _draw_session_protocol(ax: Any) -> None:
    """One identification session and the start of the next (schematic, seconds).

    The agent's input (light shading) runs from the session start to the reset. The
    decision is the first bin with |z_1 - z_2| > 2; the reset follows after a 200-ms
    hold (dark shading), or at 2 s without a decision; the next session starts from
    the reset state and is not shaded.
    """
    t = np.linspace(0.0, 2.5, 700)
    t_dec, t_reset = 1.5, 1.7
    first = 2.0 * np.clip((t - 0.15) / (t_dec - 0.15), 0.0, None) ** 1.8
    second = -1.3 * np.clip((t - t_reset - 0.15) / 0.8, 0.0, None) ** 1.8
    lead = np.where(t < t_reset, np.minimum(first, 2.6), second)
    ax.axvspan(0.0, t_dec, ymin=0.06, ymax=0.8, color=PROBE_SHADE, lw=0)
    ax.axvspan(t_dec, t_reset, ymin=0.06, ymax=0.8, color=DECISION_SHADE, alpha=DECISION_ALPHA, lw=0)
    ax.plot(t, lead, color=STROKE_COLOR, lw=1.0)
    ax.axhline(2.0, color=STROKE_COLOR, lw=0.5, ls=":")
    ax.axhline(-2.0, color=STROKE_COLOR, lw=0.5, ls=":")
    ax.axvline(t_dec, color=STROKE_COLOR, lw=0.6, ls="--")
    ax.axvline(t_reset, color=STROKE_COLOR, lw=0.9)
    ax.annotate("", xy=(t_reset, -2.75), xytext=(t_dec, -2.75),
                arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(t_reset + 0.05, -2.75, "200 ms", ha="left", va="center", fontsize=_ASSET_TICK_SIZE,
            color=DECISION_TEXT)
    ax.text(0.06, -1.0, "input", ha="left", va="center", fontsize=_ASSET_TICK_SIZE, color=PROBE_TEXT)
    # Separate the labels while keeping the arrow tips at the exact event times.
    for t_mark, label, dx in ((t_dec, "decision", -0.30), (t_reset, "reset", 0.30)):
        ax.annotate(label, xy=(t_mark, 1.01), xytext=(t_mark + dx, 1.23),
                    xycoords=ax.get_xaxis_transform(), textcoords=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=_ASSET_TICK_SIZE, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color=STROKE_COLOR,
                                    mutation_scale=6, shrinkA=3.0, shrinkB=1.0))
    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-3.8, 4.4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0", "1 s", "2 s"])
    ax.set_yticks([-2, 0, 2])
    ax.set_ylabel(r"lead $z_1-z_2$", fontsize=_ASSET_LABEL_SIZE)
    ax.set_xlabel("Time")
    ax.set_title("Session", fontsize=_ASSET_LABEL_SIZE, pad=22.0)
    style_experiment_axis(ax)


def _draw_force_timeline(ax: Any) -> None:
    """Force task session (3 s across the panel): evidence, control window, success rule."""
    from matplotlib.patches import Rectangle

    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(0, 6)
    ax.axis("off")
    scale = 10.0 / 3.0
    y = 3.4
    ax.plot([0, 10], [y, y], color=STROKE_COLOR, lw=0.6)
    ax.add_patch(Rectangle((0, y + 0.15), 1.0 * scale, 0.45, facecolor=NEUTRAL_LIGHT, edgecolor="none"))
    ax.text(0.5 * scale, y + 0.37, "evidence 6 pA", ha="center", va="center", fontsize=_ASSET_TICK_SIZE)
    ax.add_patch(Rectangle((0, y - 0.65), 1.0 * scale, 0.45, facecolor=CONTROL_SHADE, edgecolor="none"))
    ax.text(0.5 * scale, y - 0.43, "input (1 s)", ha="center", va="center", fontsize=_ASSET_TICK_SIZE)
    ax.text(2.0 * scale, y - 1.05, "success: other pool\nmakes the first decision (2 s)", ha="center", va="top",
            fontsize=_ASSET_TICK_SIZE)
    for t in (0, 1, 2, 3):
        ax.text(t * scale, 0.05, f"{t} s", ha="center", va="bottom", fontsize=_ASSET_TICK_SIZE, color=STROKE_COLOR)
    ax.set_title("Force task", fontsize=_ASSET_LABEL_SIZE, pad=2.0)


# ------------------------------------------------------------------ figure 1: identification
def _example_run(experiment_dir: Path, policy: str = "adaptive", seed: int = 0) -> dict[str, np.ndarray]:
    """Spike counts, network latent, filtered state, input, and session ends of one run."""
    import pickle

    run = experiment_dir / "tracks" / EXP_ID / policy / f"seed_{seed}" / "repeat_01"
    with open(sorted(glob.glob(str(run / "*" / "rollouts" / "*.pkl")))[0], "rb") as fh:
        rollout = pickle.load(fh)
    rows = _read(run / "state_action_trace.csv")
    return {
        "counts": rollout["obs"].reshape(len(rows), -1).numpy(),
        "z": np.array([[float(r["true_z0"]), float(r["true_z1"])] for r in rows]),
        "m": np.array([[float(r["model_z0"]), float(r["model_z1"])] for r in rows]),
        "u": np.array([[float(r["env_action_x"]), float(r["env_action_v"])] for r in rows]),
        "session_end": np.array([r["session_end"] == "True" for r in rows]),
    }


def _session_bounds(session_end: np.ndarray) -> list[tuple[int, int]]:
    """Bin ranges ``[start, stop)`` of the sessions of one run; ``session_end`` (T,) marks each reset bin."""
    ends = np.flatnonzero(session_end) + 1
    return list(zip(np.r_[0, ends[:-1]].tolist(), ends.tolist()))


def _filter_rmse(z: np.ndarray, m: np.ndarray) -> float:
    """Root-mean-square difference between network latent ``z`` and filtered mean ``m``, both (T, 2)."""
    return float(np.sqrt(np.mean((z - m) ** 2)))


def generate_identification(experiment_dir: Path, output: Path, *,
                            sessions_shown: tuple[int, ...] = (1, 20)) -> Path:
    """Identification figure: (A) circuit above (B) session structure in the first column,
    then (C) sessions ``sessions_shown`` (1-based) of PALDI seed 0 and (D) held-out R2
    along identification.

    The shown sessions follow a fixed rule (first and last), not a selection by eye.
    """
    from experiments.tnsre.eval_spiking_sessions import PROTOCOL

    r2_rows = _read(experiment_dir / "eval" / "r2_per_session.csv")
    fit_r2 = float(json.loads(Path(PROTOCOL["reduced_fit"]).read_text())["r2_test_piecewise"])
    ex = _example_run(experiment_dir)

    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.7))
    # All columns share one top and one bottom line; the legend runs above them.
    top, bottom = 0.8, 0.15
    left = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.0], left=0.065, right=0.27, top=top, bottom=bottom,
                            hspace=0.29)
    row = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.05], left=0.335, right=0.99, top=top, bottom=bottom,
                           wspace=0.3)

    ax = fig.add_subplot(left[0])
    _draw_circuit(ax)
    ax_b = fig.add_subplot(left[1])
    # Reserve space below A's footer for B's title, event labels, and arrows.
    pos_b = ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_b.y0, pos_b.width, 0.49 / fig.get_size_inches()[1]])
    _draw_session_protocol(ax_b)
    fig.text(pos_b.x0 - 22.0 / 72.0 / FIGURE_WIDTH,
             ax_b.get_position().y1 + 22.0 / 72.0 / fig.get_size_inches()[1],
             "B", ha="left", va="bottom", fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold")
    # Use the same top baseline as C and D, even though the equal-aspect circuit
    # occupies only part of its grid cell horizontally.
    fig.text(ax_b.get_position().x0 - 22.0 / 72.0 / FIGURE_WIDTH, top + 3.0 / 72.0 / 2.7,
             "A", ha="left", va="bottom", fontsize=_ASSET_PANEL_LABEL_SIZE, fontweight="bold")

    # (C) one PALDI run: raster, latent, input in the first and the last session.
    # Column widths follow session length, so all columns share one time scale.
    bounds = [_session_bounds(ex["session_end"])[k - 1] for k in sessions_shown]
    sub = row[0].subgridspec(3, len(bounds), height_ratios=[0.8, 1.25, 0.95], hspace=0.12, wspace=0.1,
                             width_ratios=[b - a for a, b in bounds])
    n_obs = ex["counts"].shape[1]
    shared_y = []
    for col, (k, (a, b)) in enumerate(zip(sessions_shown, bounds)):
        t_ms = np.arange(b - a) * 5.0
        ax_r = fig.add_subplot(sub[0, col])
        ax_z = fig.add_subplot(sub[1, col], sharex=ax_r)
        ax_u = fig.add_subplot(sub[2, col], sharex=ax_r)
        if shared_y:
            for ax, first in zip((ax_r, ax_z, ax_u), shared_y):
                ax.sharey(first)
        else:
            shared_y = [ax_r, ax_z, ax_u]
        counts = ex["counts"][a:b]
        for j in range(n_obs):
            spikes = np.flatnonzero(counts[:, j] > 0)
            ax_r.scatter(t_ms[spikes], np.full(spikes.size, j), s=2.0, marker="|", linewidths=0.6,
                         color=POOL_COLORS[0 if j < n_obs // 2 else 1], rasterized=True)
        rmse = _filter_rmse(ex["z"][a:b], ex["m"][a:b])
        ax_r.set_title(f"Session {k}", fontsize=_ASSET_LABEL_SIZE, pad=2.0, loc="left")
        ax_r.set_title(f"filter RMSE {rmse:.2f}", fontsize=_ASSET_TICK_SIZE, pad=2.5, loc="right",
                       color=STROKE_COLOR)
        ax_r.set_ylim(-1, n_obs)
        ax_r.set_yticks([20, 60])
        ax_r.set_yticklabels(["1", "2"])
        for i in range(2):
            ax_z.plot(t_ms, ex["z"][a:b, i], color=POOL_COLORS[i], lw=0.8)
            ax_z.plot(t_ms, ex["m"][a:b, i], color=POOL_COLORS[i], lw=0.55, ls="--", alpha=0.8)
            ax_u.plot(t_ms, ex["u"][a:b, i], color=POOL_COLORS[i], lw=0.6)
        ax_z.set_ylim(-2.4, 2.4)
        ax_u.set_ylim(-1.1, 1.1)
        # Decision (first bin with |z_1 - z_2| > gap, dashed) and the hold until the reset (shaded).
        lead = ex["z"][a:b, 0] - ex["z"][a:b, 1]
        decided = np.flatnonzero(np.abs(lead) > float(PROTOCOL["decision_gap"]))
        if decided.size:
            for ax in (ax_r, ax_z, ax_u):
                ax.axvspan(0.0, decided[0] * 5.0, color=PROBE_SHADE, lw=0, zorder=0)
                ax.axvspan(decided[0] * 5.0, (b - a) * 5.0, color=DECISION_SHADE,
                           alpha=DECISION_ALPHA, lw=0, zorder=0)
                ax.axvline(decided[0] * 5.0, color=STROKE_COLOR, lw=0.6, ls="--")
        # Ticks every 500 ms, none at the right edge where the next column starts.
        ax_u.set_xlim(0.0, (b - a) * 5.0)
        ax_u.set_xticks(np.arange(0.0, (b - a) * 5.0 - 150.0, 500.0))
        ax_u.set_xlabel("Session time (ms)")
        for ax in (ax_r, ax_z, ax_u):
            style_experiment_axis(ax)
            if col > 0:
                plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax_r.get_xticklabels(), visible=False)
        plt.setp(ax_z.get_xticklabels(), visible=False)
        if col == 0:
            for axis, label in zip((ax_r, ax_z, ax_u), ("Pool", "latent", "input")):
                axis.tick_params(axis="y", pad=1.0)
                axis.set_ylabel(label, fontsize=_ASSET_LABEL_SIZE, labelpad=1.0)
            _panel_label(ax_r, "C", dx=-34)
            ax_z.plot([], [], color=STROKE_COLOR, lw=0.8, label="network")
            ax_z.plot([], [], color=STROKE_COLOR, lw=0.55, ls="--", label="filtered")
            ax_z.legend(loc="upper left", fontsize=_ASSET_TICK_SIZE, frameon=False, ncol=2,
                        handlelength=1.4, borderaxespad=0.1)
    fig.align_ylabels(shared_y)
    # (D) held-out R2 after each identification session.
    ax = fig.add_subplot(row[1])
    by = _r2_by_session(r2_rows)
    for policy in POLICIES:
        ks = np.array(sorted(k for p, k in by if p == policy))
        vals = [np.asarray(by[(policy, k)]) for k in ks]
        color = _asset_baseline_policy_color(policy)
        ax.plot(ks, [np.median(v) for v in vals], color=color, lw=0.95)
        ax.fill_between(ks, [np.percentile(v, 25) for v in vals], [np.percentile(v, 75) for v in vals],
                        color=color, alpha=0.1, lw=0)
    ax.axhline(fit_r2, color=REFERENCE_STYLE["reduced_fit"]["color"], ls=":", lw=0.8)
    ax.set_xlim(0, FINAL_SESSIONS)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Identification sessions")
    ax.set_ylabel(r"Held-out $R^2_{\mathrm{roll}}$")
    ax.set_title("Prediction", fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    style_experiment_axis(ax)
    _panel_label(ax, "D", dx=-32)

    handles = [plt.Line2D([], [], color=_asset_baseline_policy_color(p), lw=1.0, label=_asset_policy_label(p))
               for p in POLICIES]
    handles.append(plt.Line2D([], [], color=REFERENCE_STYLE["reduced_fit"]["color"], ls=":", lw=0.9,
                              label=REFERENCE_STYLE["reduced_fit"]["label"]))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=_ASSET_TICK_SIZE, columnspacing=0.9, handlelength=1.4, handletextpad=0.3, frameon=False)
    return save_figure(fig, output, plt_module=plt)


# ------------------------------------------------------------------ figure 2: control
def _draw_budget_panel(ax: Any, look: dict, task: str, budgets: list[float], *, ylabel: bool) -> None:
    """Task success after the final identification session versus the energy budget."""
    for policy in POLICIES:
        pts = [look.get((policy, FINAL_SESSIONS, task, b), (np.nan, 0.0)) for b in budgets]
        m, e = (np.array(v) for v in zip(*pts))
        ax.errorbar(budgets, m, yerr=e, color=_asset_baseline_policy_color(policy), lw=0.9, marker="o", ms=2.2,
                    capsize=1.2, elinewidth=0.6)
    for name, style in REFERENCE_STYLE.items():
        vals = [look.get((name, -1, task, b), (np.nan, 0.0))[0] for b in budgets]
        ax.plot(budgets, vals, lw=0.8, ms=2.4, **style)
    pad = 0.15 * (budgets[-1] - budgets[0])
    ax.set_xlim(budgets[0] - pad, budgets[-1] + pad)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b:g}" for b in budgets])
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"After {FINAL_SESSIONS} sessions", fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    ax.set_xlabel(r"Energy budget $\int\|u\|^2dt$")
    if ylabel:
        ax.set_ylabel("Task success")
    style_experiment_axis(ax)


def _draw_session_panel(ax: Any, look: dict, task: str, checkpoints: list[int], *, ylabel: bool) -> None:
    """Task success at budget ``CURVE_BUDGETS[task]`` along identification (0 = shared initial estimate)."""
    pos = np.arange(len(checkpoints))
    b = CURVE_BUDGETS[task]
    prior = look.get(("prior", 0, task, b), (np.nan, 0.0))
    for policy in POLICIES:
        pts = [prior if k == 0 else look.get((policy, k, task, b), (np.nan, 0.0)) for k in checkpoints]
        m, e = (np.array(v) for v in zip(*pts))
        color = _asset_baseline_policy_color(policy)
        ax.plot(pos, m, color=color, lw=0.9, marker="o", ms=2.2)
        ax.fill_between(pos, m - e, m + e, color=color, alpha=0.1, lw=0)
    for name, style in REFERENCE_STYLE.items():
        ax.axhline(look.get((name, -1, task, b), (np.nan, 0.0))[0], color=style["color"], ls=style["linestyle"],
                   lw=0.8)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in checkpoints])
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"Budget {b:g}", fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    ax.set_xlabel("Identification sessions")
    if ylabel:
        ax.set_ylabel("Task success")
    style_experiment_axis(ax)


def _draw_example_session(fig: Any, cell: Any, eval_dir: Path, task: str) -> Any:
    """Example session of ``task``: task timeline (top), target lead under each controller,
    and PALDI's input, on one time axis (ms) cut at ``T_SHOWN_MS``. Circles mark decisions:
    the uncontrolled decision (shared by all controllers) and each successful overturn.
    Returns the timeline axis."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    from experiments.tnsre.eval_spiking_sessions import EXAMPLES, PROTOCOL

    ex = dict(np.load(eval_dir / "examples" / "example_sessions.npz"))
    ex_rows = {(r["task"], r["controller"]): r for r in _read(eval_dir / "examples" / "example_sessions.csv")}
    _seed, budget = EXAMPLES["sessions"][task]
    sub = cell.subgridspec(3, 1, height_ratios=[0.8, 1.5, 1.0], hspace=0.1)
    ax_t = fig.add_subplot(sub[0])
    ax_l = fig.add_subplot(sub[1], sharex=ax_t)
    ax_u = fig.add_subplot(sub[2], sharex=ax_t)
    free = EXAMPLES["model_free"][task]
    # Drawn in this order: no input as a wide pale baseline underneath, the fit dotted on top of PALDI.
    styles = {"none": dict(color=NEUTRAL_LIGHT, lw=1.8, label="no input"),
              free: dict(color=REFERENCE_STYLE[free]["color"], lw=0.9, ls=REFERENCE_STYLE[free]["linestyle"],
                         label=REFERENCE_STYLE[free]["label"].replace(" from onset", "")),
              "learned": dict(color=_asset_baseline_policy_color("adaptive"), lw=1.0, label="PALDI model"),
              "fit": dict(color=REFERENCE_STYLE["reduced_fit"]["color"], lw=1.0, ls=":", label="fitted model")}
    onset_ms_bin = int(ex_rows[(task, "learned")]["onset"])
    onset_ms = onset_ms_bin * 5.0
    target = int(ex_rows[(task, "learned")]["target"])
    evidence_ms = float(PROTOCOL["evidence_bins"]) * 5.0
    window_ms = float(PROTOCOL["window"][task]) * 5.0

    # Timeline of this session: evidence and control window; the success window ends below.
    ax_t.add_patch(Rectangle((0.0, 1.275), evidence_ms, 0.95, facecolor=NEUTRAL_LIGHT, edgecolor="none"))
    ax_t.text(20.0, 1.75, "evidence 6 pA", ha="left", va="center", fontsize=_ASSET_TICK_SIZE)
    ax_t.add_patch(Rectangle((onset_ms, 0.075), window_ms, 0.95, facecolor=CONTROL_SHADE,
                             edgecolor="none"))
    ax_t.text(onset_ms + 0.5 * window_ms, 0.55, "control input (at most 2 s)", ha="center", va="center",
              fontsize=_ASSET_TICK_SIZE)
    ax_t.set_ylim(0.0, 2.4)
    ax_t.axis("off")

    for name, st in styles.items():
        z = ex[f"{task}_{name}_z"]
        lead = z[:, target] - z[:, 1 - target]
        ax_l.plot(np.arange(z.shape[0]) * 5.0, lead, **st)
        # A successful session ends in the bin where the target pool's decision is made.
        if int(ex_rows[(task, name)]["success"]):
            ax_l.plot((z.shape[0] - 1) * 5.0, lead[-1], ls="none", **DECISION_MARKER,
                      markerfacecolor=st["color"])
    # The uncontrolled decision ends the evidence period (bin onset - 1), the same in all traces.
    z = ex[f"{task}_learned_z"]
    ax_l.plot(onset_ms - 5.0, z[onset_ms_bin - 1, target] - z[onset_ms_bin - 1, 1 - target], ls="none",
              **DECISION_MARKER, markerfacecolor=STROKE_COLOR)
    u = ex[f"{task}_learned_u"]
    t_ms = np.arange(u.shape[0]) * 5.0
    ax_u.plot(t_ms, u[:, target], color=POOL_COLORS[1], lw=0.8, label="to losing pool")
    ax_u.plot(t_ms, u[:, 1 - target], color=POOL_COLORS[0], lw=0.8, label="to winning pool")
    ax_t.axvline(onset_ms, ymax=0.94, color=STROKE_COLOR, lw=0.6, ls="--")
    for a in (ax_l, ax_u):
        a.axvline(onset_ms, color=STROKE_COLOR, lw=0.6, ls="--")
        style_experiment_axis(a)
    ax_l.axhline(2.0, color=STROKE_COLOR, lw=0.5, alpha=0.6)
    ax_l.set_ylabel("lead", fontsize=_ASSET_LABEL_SIZE)
    ax_t.set_title(f"One session (budget {budget:g})", fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    handles, labels = ax_l.get_legend_handles_labels()
    handles = [handles[labels.index("no input")], Line2D([], [], ls="none", markerfacecolor=STROKE_COLOR,
                                                         label="decision", **DECISION_MARKER)]
    ax_l.legend(handles=handles, loc="upper left", fontsize=_ASSET_TICK_SIZE, frameon=False, handlelength=1.4,
                ncol=2, borderaxespad=0.1, columnspacing=0.8)
    ax_l.set_ylim(-3.6, 3.6)
    ax_l.set_yticks([-2, 0, 2])
    for label in ax_l.get_xticklabels():
        label.set_visible(False)
    ax_u.set_ylabel("input", fontsize=_ASSET_LABEL_SIZE)
    ax_u.set_ylim(-1.1, 1.1)
    ax_u.set_xlabel("Time (ms)")
    ax_u.set_xlim(0.0, T_SHOWN_MS)
    # PALDI's input is zero before the decision; its legend sits in that span.
    # Direct labels at the end of PALDI's input, where the traces separate.
    for series, name, color in ((u[:, target], "losing", POOL_COLORS[1]), (u[:, 1 - target], "winning", POOL_COLORS[0])):
        ax_u.text(t_ms[-1] + 40.0, series[-1], name, ha="left", va="center", fontsize=_ASSET_TICK_SIZE, color=color)
    return ax_t


def _control_legend(fig: Any, plt: Any) -> None:
    reference_labels = {"reduced_fit": "Fitted", "spread": "Uniform", "front": "Full"}
    handles = [plt.Line2D([], [], color=_asset_baseline_policy_color(p), lw=1.0, label=_asset_policy_label(p))
               for p in POLICIES]
    handles += [plt.Line2D([], [], color=s["color"], ls=s["linestyle"], marker=s["marker"], ms=2.4, lw=0.8,
                           label=reference_labels[name]) for name, s in REFERENCE_STYLE.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=_ASSET_TICK_SIZE, columnspacing=0.9, handlelength=1.4, handletextpad=0.3, frameon=False)


def generate_control(experiment_dir: Path, output: Path, *, summary_name: str = "task_summary_warm.csv") -> Path:
    """Overturn-task figure of the main text: (A) success vs budget, (B) success along
    identification, (C) task timeline over one example session. Warm-start controller
    unless ``summary_name`` says otherwise."""
    from experiments.tnsre.eval_spiking_sessions import PROTOCOL

    eval_dir = experiment_dir / "eval"
    look = _summary_lookup(eval_dir / summary_name)
    checkpoints = [int(k) for k in PROTOCOL["task_checkpoints"]]
    task = "overturn"

    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.05))
    row = fig.add_gridspec(1, 3, width_ratios=[0.5, 1.0, 2.15], left=0.065, right=0.985, top=0.8,
                           bottom=0.2, wspace=0.3)
    ax = fig.add_subplot(row[0])
    _draw_budget_panel(ax, look, task, [float(b) for b in PROTOCOL["budgets"][task]], ylabel=True)
    _panel_label(ax, "A", dx=-28)  # left of the tick labels: the title is wider than this narrow panel
    ax = fig.add_subplot(row[1], sharey=ax)
    _draw_session_panel(ax, look, task, checkpoints, ylabel=False)
    _panel_label(ax, "B")
    _panel_label(_draw_example_session(fig, row[2], eval_dir, task), "C", dx=-30)
    _control_legend(fig, plt)
    return save_figure(fig, output, plt_module=plt)


def generate_force(experiment_dir: Path, output: Path, *, summary_name: str = "task_summary_warm.csv") -> Path:
    """Force-task figure of the Supplementary: (A) task, (B) success vs budget, (C) success along identification."""
    from experiments.tnsre.eval_spiking_sessions import PROTOCOL

    look = _summary_lookup(experiment_dir / "eval" / summary_name)
    checkpoints = [int(k) for k in PROTOCOL["task_checkpoints"]]
    task = "force"

    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.05))
    row = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.0, 1.0], left=0.03, right=0.985, top=0.78, bottom=0.2,
                           wspace=0.4)
    ax = fig.add_subplot(row[0])
    _draw_force_timeline(ax)
    _panel_label(ax, "A", dx=0)
    ax = fig.add_subplot(row[1])
    _draw_budget_panel(ax, look, task, [float(b) for b in PROTOCOL["budgets"][task]], ylabel=True)
    _panel_label(ax, "B")
    ax = fig.add_subplot(row[2])
    _draw_session_panel(ax, look, task, checkpoints, ylabel=False)
    _panel_label(ax, "C")
    _control_legend(fig, plt)
    return save_figure(fig, output, plt_module=plt)


# ------------------------------------------------------------------ figure 3: diagnosis
def _parse_refits(path: Path) -> list[dict[str, float]]:
    """Batch refits on a run's own session data, from ``refit_on_session_data.txt``."""
    out = []
    for line in path.read_text().splitlines():
        m = re.search(r"refit from \w+\s+R2 ([0-9.]+)\s+(\{.*\})", line)
        if m:
            out.append({"r2": float(m.group(1)), **ast.literal_eval(m.group(2))})
    return out


def generate_diagnosis(experiment_dir: Path, output: Path) -> Path:
    from experiments.tnsre.eval_spiking_sessions import reference_fit

    eval_dir = experiment_dir / "eval"
    diag_dir = experiment_dir / "diagnostics" / "control_gap"
    regime = _read(diag_dir / "control_regime_r2.csv")
    swap = {r["variant"]: (float(r["success"]), float(r["sem_over_seeds"])) for r in _read(diag_dir / "summary.csv")}
    refits = _parse_refits(diag_dir / "refit_on_session_data.txt")
    r2 = {(r["policy"], int(r["seed"]), int(r["sessions"])): (float(r["r2"]), [float(x) for x in r["theta"].split()])
          for r in _read(eval_dir / "r2_per_session.csv")}
    success = _per_seed_success(eval_dir / "task", "overturn", 12.0)
    fit = reference_fit()

    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.35, 1.0, 1.0], left=0.065, right=0.985, top=0.83,
                          bottom=0.2, wspace=0.7)

    # (A) control-regime R2 by push direction.
    ax = fig.add_subplot(gs[0])
    directions = ("excite", "both", "suppress")
    models = [("fit", -1)] + [(p, FINAL_SESSIONS) for p in ACTIVE]
    width = 0.8 / len(models)
    for j, (policy, k) in enumerate(models):
        vals = [[float(r["r2"]) for r in regime if r["direction"] == d and r["policy"] == policy
                 and int(r["sessions"]) == k] for d in directions]
        med = [np.median(v) for v in vals]
        lo = [np.median(v) - np.percentile(v, 25) for v in vals]
        hi = [np.percentile(v, 75) - np.median(v) for v in vals]
        color = REFERENCE_STYLE["reduced_fit"]["color"] if policy == "fit" else _asset_baseline_policy_color(policy)
        ax.bar(np.arange(3) + (j - (len(models) - 1) / 2) * width, med, width * 0.9, yerr=[lo, hi], color=color,
               error_kw=dict(lw=0.5, capsize=1.0))
    ax.set_xticks(range(3))
    ax.set_xticklabels(["excite\nlosing pool", "both", "suppress\nwinning pool"], fontsize=_ASSET_TICK_SIZE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(r"$R^2_{\mathrm{roll}}$, control regime")
    style_experiment_axis(ax)
    _panel_label(ax, "A")

    # (B) parameter swaps and the planner/filter split.
    ax = fig.add_subplot(gs[1])
    names = {"w_plus": r"$w_+$", "w_minus": r"$w_-$", "h_raw": r"$h$", "gamma_raw": r"$\gamma$", "g_raw": r"$g$"}
    order = ([("learned", "PALDI estimate"), ("fit", "fitted model")]
             + [(f"learned+fit:{p}", f"PALDI, fit {n}") for p, n in names.items()]
             + [(f"fit+learned:{p}", f"fit, PALDI {n}") for p, n in names.items()]
             + [("plan:learned/filter:fit", "plan PALDI, filter fit"),
                ("plan:fit/filter:learned", "plan fit, filter PALDI")])
    for i, (key, _label) in enumerate(order):
        m, e = swap[key]
        color = (_asset_baseline_policy_color("adaptive") if key.startswith(("learned", "plan:learned"))
                 else REFERENCE_STYLE["reduced_fit"]["color"])
        ax.barh(i, m, xerr=e, height=0.65, color=color, alpha=0.9 if key in ("learned", "fit") else 0.55,
                error_kw=dict(lw=0.5, capsize=1.0))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab for _, lab in order], fontsize=_ASSET_TICK_SIZE)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Overturn success (budget 12)")
    style_experiment_axis(ax)
    _panel_label(ax, "B", dx=-55)

    # (C) held-out R2 versus overturn success per seed.
    ax = fig.add_subplot(gs[2])
    rng = np.random.default_rng(0)  # vertical jitter only, for overlapping success fractions
    all_x, all_y = [], []
    for policy in ACTIVE:
        pts = [(r2[(policy, s, k)][0], success[(policy, s, k)]) for (p, s, k) in success
               if p == policy and k in (1, 5, 20) and (policy, s, k) in r2]
        x, y = (np.array(v) for v in zip(*pts))
        all_x.append(x)
        all_y.append(y)
        ax.scatter(x, y + rng.uniform(-0.015, 0.015, y.size), s=5, color=_asset_baseline_policy_color(policy),
                   alpha=0.6, lw=0)
    r = float(np.corrcoef(np.concatenate(all_x), np.concatenate(all_y))[0, 1])
    ax.text(0.03, 0.97, f"r = {r:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=_ASSET_TICK_SIZE)
    # Early checkpoints have negative R2; show every point that enters r.
    ax.set_xlim(min(0.0, np.floor(np.concatenate(all_x).min() * 10) / 10), 1.0)
    ax.set_xticks(np.arange(np.ceil(ax.get_xlim()[0] * 2) / 2, 1.01, 0.5))
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r"Held-out $R^2_{\mathrm{roll}}$")
    ax.set_ylabel("Overturn success (budget 12)")
    style_experiment_axis(ax)
    _panel_label(ax, "C")

    # (D) learned cross-inhibition and gain scale.
    ax = fig.add_subplot(gs[3])
    for policy in ACTIVE:
        th = np.array([r2[(policy, s, FINAL_SESSIONS)][1] for s in range(20) if (policy, s, FINAL_SESSIONS) in r2])
        ax.scatter(th[:, 1], [_softplus(x) for x in th[:, 3]], s=6, color=_asset_baseline_policy_color(policy),
                   alpha=0.75, lw=0)
    if refits:
        ax.scatter([r["w_minus"] for r in refits], [r["gamma"] for r in refits], s=14, marker="x", lw=0.7,
                   color=STROKE_COLOR, label="refit on session data")
    ax.scatter([fit[1]], [_softplus(fit[3])], s=28, marker="s", color=REFERENCE_STYLE["reduced_fit"]["color"],
               label="fitted model", zorder=5)
    ax.set_xlabel(r"cross-inhibition $w_-$")
    ax.set_ylabel(r"gain scale $\gamma$")
    ax.legend(loc="upper right", fontsize=_ASSET_TICK_SIZE, frameon=False, handletextpad=0.2, borderaxespad=0.1)
    style_experiment_axis(ax)
    _panel_label(ax, "D")

    handles = [plt.Line2D([], [], color=_asset_baseline_policy_color(p), lw=1.0, label=_asset_policy_label(p))
               for p in ACTIVE]
    handles.append(plt.Line2D([], [], color=REFERENCE_STYLE["reduced_fit"]["color"], lw=3, label="fitted model"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=_ASSET_TICK_SIZE, columnspacing=0.9, handlelength=1.4, handletextpad=0.3, frameon=False)
    return save_figure(fig, output, plt_module=plt)


FIGURES = {
    "identification": ("tnsre_fig_spiking_identification", generate_identification),
    "control": ("tnsre_fig_spiking_control", generate_control),
    "force": ("tnsre_fig_spiking_force", generate_force),
    "diagnosis": ("tnsre_fig_spiking_diagnosis", generate_diagnosis),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Figures of the spiking-network decision experiment.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--figure", choices=[*FIGURES, "all"], default="all")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <experiment-dir>/eval/figures")
    parser.add_argument("--format", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args(argv)
    out_dir = args.out_dir or args.experiment_dir / "eval" / "figures"
    names = list(FIGURES) if args.figure == "all" else [args.figure]
    for name in names:
        stem, fn = FIGURES[name]
        print(fn(args.experiment_dir, out_dir / f"{stem}.{args.format}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
