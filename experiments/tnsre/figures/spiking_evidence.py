"""Identification with and without sensory evidence in the decision sessions (spiking network, M2 learner).

Compares runs of the same 6 policies x 20 seeds, paired by seed (same network seeds and readouts),
each scored by ``experiments.tnsre.eval_spiking_sessions`` on the same held-out trajectories and
task sessions. The first ``--run`` is the reference condition (no evidence,
``wong_wang_snn_sessions_m2``); it also holds the fitted-model and model-free task references.
Later runs are evidence variants (``wong_wang_snn_sessions_m2_evidence``: 6 pA to a random pool
for the first 1 s of every session; ``..._evidence_u2``: the same with input bound 2).

Panels: (A) held-out rollout R2 after each identification session, active policies (median over
seeds), (B) R2 after 20 sessions (median and IQR), (C, D) overturn success at budget 12 and force
success at budget 1 after 20 sessions (warm-start controller; mean and SEM over seeds; gray lines:
fitted model and the best model-free push), (E) R2 after 20 sessions on recorded overturn trials
whose push suppresses the winning pool (``diag_control_gap`` recordings; dotted: fitted model).

Usage:
    python -m experiments.tnsre.figures.spiking_evidence \\
        --run "without evidence=results/tnsre/20260924_snn_sessions_m2" \\
        --run "with evidence=results/tnsre/20260928_snn_sessions_m2_evidence" \\
        --run "with evidence, |u| <= 2=results/tnsre/20260929_snn_sessions_m2_evidence_u2" \\
        --out results/tnsre/20260929_snn_sessions_m2_evidence_u2/eval/figures/tnsre_fig_spiking_evidence.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from actdyn.utils.figure_io import load_plotting

from .assets import (
    _ASSET_LABEL_SIZE,
    _ASSET_TICK_SIZE,
    _apply_asset_style,
    _asset_baseline_policy_color,
    _asset_policy_label,
    save_figure,
)
from .spiking_sessions import (
    ACTIVE,
    FIGURE_WIDTH,
    FINAL_SESSIONS,
    POLICIES,
    REFERENCE_STYLE,
    _panel_label,
    _r2_by_session,
    _read,
)
from .theme import STROKE_COLOR, style_experiment_axis

# Line style, marker, and marker face (None = filled) of the runs, in --run order.
RUN_STYLES = (("--", "o", "white"), ("-", "o", None), (":", "s", None))
TASK_PANELS = (("overturn", 12.0, "front"), ("force", 1.0, "spread"))
# Control-regime R2 of a run's models: the reference run keeps it with the control-gap diagnostic.
REGIME_R2 = ("diagnostics/control_regime_r2.csv", "diagnostics/control_gap/control_regime_r2.csv")


def _task_rows(eval_dir: Path, subdir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted((eval_dir / subdir).glob("worker_*.csv")):
        rows.extend(_read(path))
    return [r for r in rows if r["valid"] == "1"]


def _seed_success(rows: list[dict[str, str]], policy: str, checkpoint: int, task: str, budget: float) -> np.ndarray:
    """Mean success per training seed over the valid task sessions, shape (n_seeds,)."""
    per_seed: dict[int, list[int]] = {}
    for r in rows:
        if (r["policy"] == policy and int(r["checkpoint"]) == checkpoint and r["task"] == task
                and float(r["budget"]) == budget):
            per_seed.setdefault(int(r["seed"]), []).append(int(r["success"]))
    return np.array([np.mean(per_seed[s]) for s in sorted(per_seed)])


def _regime_rows(run_dir: Path) -> list[dict[str, str]]:
    path = next(run_dir / rel for rel in REGIME_R2 if (run_dir / rel).exists())
    return [r for r in _read(path) if r["direction"] == "suppress"]


def _dot(ax, xi: float, center: float, low: float, high: float, color: str, style: tuple) -> None:
    _ls, marker, face = style
    ax.plot([xi, xi], [low, high], color=color, lw=0.8)
    ax.plot(xi, center, marker=marker, ms=3.2, color=color, markerfacecolor=face or color, markeredgewidth=0.8)


def generate_evidence_comparison(runs: list[tuple[str, Path]], output: Path) -> Path:
    r2 = [_r2_by_session(_read(d / "eval" / "r2_per_session.csv")) for _l, d in runs]
    tasks = [_task_rows(d / "eval", "task_warm") for _l, d in runs]
    regime = [_regime_rows(d) for _l, d in runs]
    reference_dir = runs[0][1]
    # References: the fit with the warm controller; model-free pushes from the fresh run (they do not plan).
    free = [r for r in _task_rows(reference_dir / "eval", "task") if r["controller"] in ("none", "spread", "front")]
    references = free + [r for r in tasks[0] if r["policy"] == "reduced_fit"]
    offsets = (np.arange(len(runs)) - 0.5 * (len(runs) - 1)) * 0.26
    x = np.arange(len(POLICIES))

    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.3))
    row = fig.add_gridspec(1, 5, width_ratios=[1.3, 1.0, 1.0, 1.0, 1.0], left=0.065, right=0.99, top=0.76,
                           bottom=0.24, wspace=0.55)

    # (A) held-out R2 along identification, active policies.
    ax = fig.add_subplot(row[0])
    for (ls, _m, _f), table in zip(RUN_STYLES, r2):
        for policy in ACTIVE:
            ks = np.array(sorted(k for p, k in table if p == policy))
            ax.plot(ks, [np.median(table[(policy, k)]) for k in ks], color=_asset_baseline_policy_color(policy),
                    lw=0.9, ls=ls)
    ax.set_xlim(0, FINAL_SESSIONS)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Identification sessions")
    ax.set_ylabel(r"Held-out $R^2_{\mathrm{roll}}$")
    ax.set_title("Prediction", fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    style_experiment_axis(ax)
    _panel_label(ax, "A", dx=-30)

    def policy_axis(ax, title: str, letter: str, dx: float = -7.2) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels([_asset_policy_label(p) for p in POLICIES], rotation=40, ha="right")
        ax.set_title(title, fontsize=_ASSET_LABEL_SIZE, pad=2.0)
        style_experiment_axis(ax)
        _panel_label(ax, letter, dx=dx)

    # (B) R2 after 20 sessions.
    ax = fig.add_subplot(row[1])
    for style, off, table in zip(RUN_STYLES, offsets, r2):
        for i, policy in enumerate(POLICIES):
            v = np.asarray(table[(policy, FINAL_SESSIONS)])
            _dot(ax, x[i] + off, np.median(v), np.percentile(v, 25), np.percentile(v, 75),
                 _asset_baseline_policy_color(policy), style)
    ax.set_ylim(-0.2, 1.0)
    policy_axis(ax, f"After {FINAL_SESSIONS} sessions", "B")

    # (C, D) task success after 20 sessions.
    for n, (task, budget, push) in enumerate(TASK_PANELS):
        ax = fig.add_subplot(row[2 + n])
        for style, off, rows in zip(RUN_STYLES, offsets, tasks):
            for i, policy in enumerate(POLICIES):
                v = _seed_success(rows, policy, FINAL_SESSIONS, task, budget)
                sem = v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else 0.0
                _dot(ax, x[i] + off, v.mean(), v.mean() - sem, v.mean() + sem, _asset_baseline_policy_color(policy),
                     style)
        for name in ("reduced_fit", push):
            v = _seed_success(references, name, -1, task, budget)
            ax.axhline(v.mean(), color=REFERENCE_STYLE[name]["color"], ls=REFERENCE_STYLE[name]["linestyle"], lw=0.8)
        ax.set_ylim(-0.03, 1.03)
        if n == 0:
            ax.set_ylabel("Task success")
        policy_axis(ax, f"{task.capitalize()} (budget {budget:g})", "CD"[n], dx=-22 if n == 0 else -7.2)

    # (E) prediction when the push suppresses the winning pool.
    ax = fig.add_subplot(row[4])
    for style, off, rows in zip(RUN_STYLES, offsets, regime):
        for i, policy in enumerate(POLICIES):
            v = np.array([float(r["r2"]) for r in rows if r["policy"] == policy and int(r["sessions"]) == FINAL_SESSIONS])
            _dot(ax, x[i] + off, np.median(v), np.percentile(v, 25), np.percentile(v, 75),
                 _asset_baseline_policy_color(policy), style)
    fit_r2 = [float(r["r2"]) for r in regime[0] if r["policy"] == "fit"]
    ax.axhline(fit_r2[0], color=REFERENCE_STYLE["reduced_fit"]["color"], ls=":", lw=0.8)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel(r"$R^2_{\mathrm{roll}}$", labelpad=0.5)
    policy_axis(ax, "Suppress push", "E", dx=-22)

    handles = [plt.Line2D([], [], color=STROKE_COLOR, lw=0.9, ls=ls, marker=marker, ms=3.2,
                          markerfacecolor=face or STROKE_COLOR, label=label)
               for (label, _d), (ls, marker, face) in zip(runs, RUN_STYLES)]
    handles += [plt.Line2D([], [], color=_asset_baseline_policy_color(p), lw=1.0, label=_asset_policy_label(p))
                for p in POLICIES]
    handles += [plt.Line2D([], [], color=REFERENCE_STYLE[name]["color"], ls=REFERENCE_STYLE[name]["linestyle"], lw=0.8,
                           label=REFERENCE_STYLE[name]["label"]) for name in ("reduced_fit", "front", "spread")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=6, fontsize=_ASSET_TICK_SIZE,
               columnspacing=0.9, handlelength=1.8, handletextpad=0.3, frameon=False)
    return save_figure(fig, output, plt_module=plt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identification with and without session evidence.")
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                        help="result root and its legend label; the first is the no-evidence reference")
    parser.add_argument("--out", type=Path, required=True, help="output file (.pdf, .png, or .svg)")
    args = parser.parse_args(argv)
    runs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.run]
    if len(runs) > len(RUN_STYLES):
        raise ValueError(f"at most {len(RUN_STYLES)} runs")
    print(generate_evidence_comparison(runs, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
