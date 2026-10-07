"""Figures comparing weak and strong identification control in the evidence-led SNN trials.

Inputs (no simulation): the weak experiment folder written by
``experiments.tnsre.snn_weak_evidence_finalize`` and the frozen selected result (strong control
throughout, ``results/tnsre/20261007_snn_native_uniform``). Both use the same 20 training
seeds, readouts, priors, network seeds, and task panel.

* ``spiking_weak_evidence_comparison``: (A) held-out R2 along identification for PALDI and
  RHC-US (median, IQR over 20 seeds), (B) R2 after 20 trials for all six methods, (C) overturn
  success at 480 pA^2 s along identification, (D) success after 20 trials (mean +/- SEM over
  seeds, warm-start controller, seven valid task trials).
* ``spiking_weak_evidence_trials``: trials 1-10 only, (A) RMS identification input, (B) first
  choices that flip relative to the evidence-only replay with the same network seed (both decided), (C) decision time.

Usage:
    python -m experiments.tnsre.figures.spiking_weak_evidence --experiment-dir <weak folder>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from actdyn.utils.figure_io import load_plotting

from .assets import _ASSET_LABEL_SIZE, _ASSET_TICK_SIZE, _apply_asset_style, _asset_baseline_policy_color, \
    _asset_policy_label, save_figure
from .spiking_sessions import BUDGET_PA2_S, CONTROL_CHECKPOINTS, CURVE_STYLE, FIGURE_WIDTH, POLICIES, \
    _panel_label, _shade_evidence_trials
from .theme import style_experiment_axis

FROZEN = Path('results/tnsre/20261007_snn_native_uniform')
SCHEDULES = ('strong', 'weak')
# Schedule encoding, shared by every panel: strong = solid line / open marker, weak = dashed / filled.
LINE = {'strong': '-', 'weak': '--'}
OFFSET = {'strong': -0.17, 'weak': 0.17}
BUDGET = 12.0  # normalized; 480 pA^2 s
SHOWN = ('adaptive', 'rhc')  # the two leading methods in panels A and C
WEAK_TRIALS = (1, 10)


def _fill(schedule: str, color) -> dict:
    return dict(markerfacecolor='white' if schedule == 'strong' else color, markeredgecolor=color)


def _r2(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / 'eval' / 'r2_per_session.csv')


def _success(folder: Path) -> pd.DataFrame:
    summary = pd.read_csv(folder / 'eval' / 'task_summary_warm.csv')
    return summary[(summary.task == 'overturn')]


def _method_axis(ax, ylabel: str):
    ax.set_xticks(range(len(POLICIES)), [_asset_policy_label(p) for p in POLICIES], rotation=30, ha='right',
                  rotation_mode='anchor')
    ax.set_xlim(-0.6, len(POLICIES) - 0.4)
    ax.set_ylabel(ylabel)
    style_experiment_axis(ax)


def comparison(folders: dict[str, Path], output: Path) -> Path:
    r2 = {s: _r2(f) for s, f in folders.items()}
    success = {s: _success(f) for s, f in folders.items()}
    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.1))
    grid = fig.add_gridspec(1, 4, left=0.065, right=0.995, top=0.79, bottom=0.25, wspace=0.42,
                            width_ratios=[1.0, 1.05, 1.0, 1.05])
    # A: R2 along identification; R2 after trial k sits at x = k.
    ax = fig.add_subplot(grid[0])
    _shade_evidence_trials(ax, WEAK_TRIALS[0] - 1, WEAK_TRIALS[1])
    for policy in SHOWN:
        color = _asset_baseline_policy_color(policy)
        for schedule in SCHEDULES:
            q = r2[schedule][r2[schedule].policy == policy].groupby('sessions').r2.quantile([.25, .5, .75]).unstack()
            assert (q.index == np.arange(21)).all()
            ax.plot(q.index, q[.5], color=color, lw=0.95, ls=LINE[schedule], marker=CURVE_STYLE[policy]['marker'],
                    markersize=2.2, markevery=5, markeredgewidth=.4, **_fill(schedule, color))
            ax.fill_between(q.index, q[.25], q[.75], color=color, alpha=0.08, lw=0)
    ax.set_xlim(0, 20); ax.set_ylim(-0.2, 1.0); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Identification trials'); ax.set_ylabel(r'Held-out $R^2_{\mathrm{roll}}$')
    ax.set_title('Prediction', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    style_experiment_axis(ax); _panel_label(ax, 'A', dx=-30)
    # B: R2 after 20 trials, median and IQR.
    ax = fig.add_subplot(grid[1])
    for j, policy in enumerate(POLICIES):
        color = _asset_baseline_policy_color(policy)
        for schedule in SCHEDULES:
            v = r2[schedule][(r2[schedule].policy == policy) & (r2[schedule].sessions == 20)].r2.to_numpy()
            assert len(v) == 20
            lo, med, hi = np.percentile(v, [25, 50, 75])
            ax.errorbar(j + OFFSET[schedule], med, yerr=[[med - lo], [hi - med]], color=color, lw=0.8, capsize=1.2,
                        marker=CURVE_STYLE[policy]['marker'], ms=2.8, markeredgewidth=0.5, **_fill(schedule, color))
    ax.set_ylim(-0.2, 1.0)
    ax.set_title('After 20 trials', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    _method_axis(ax, r'Held-out $R^2_{\mathrm{roll}}$'); _panel_label(ax, 'B', dx=-30)
    # C: success at 480 pA^2 s along identification (checkpoint 0 = shared prior).
    ax = fig.add_subplot(grid[2])
    pos = np.arange(len(CONTROL_CHECKPOINTS))
    inside = [i for i, k in enumerate(CONTROL_CHECKPOINTS) if WEAK_TRIALS[0] <= k <= WEAK_TRIALS[1]]
    _shade_evidence_trials(ax, inside[0] - 1, inside[-1], label_y=0.86)
    for policy in SHOWN:
        color = _asset_baseline_policy_color(policy)
        for schedule in SCHEDULES:
            s = success[schedule]
            pts = [s[(s.policy == ('prior' if k == 0 else policy)) & (s.checkpoint == k) & (s.budget == BUDGET)]
                   for k in CONTROL_CHECKPOINTS]
            assert all(len(p) == 1 for p in pts)
            m = np.array([float(p.success.iloc[0]) for p in pts]); e = np.array([float(p.sem_over_seeds.iloc[0]) for p in pts])
            ax.plot(pos, m, color=color, lw=0.95, ls=LINE[schedule], marker=CURVE_STYLE[policy]['marker'], ms=2.2,
                    markeredgewidth=.4, **_fill(schedule, color))
            ax.fill_between(pos, m - e, m + e, color=color, alpha=0.08, lw=0)
    ax.set_xticks(pos, [str(k) for k in CONTROL_CHECKPOINTS]); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Identification trials'); ax.set_ylabel('Task success')
    ax.set_title(rf'Overturn, {BUDGET * BUDGET_PA2_S:g} pA$^2$ s', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    style_experiment_axis(ax); _panel_label(ax, 'C', dx=-30)
    # D: success after 20 trials, mean +/- SEM over training seeds.
    ax = fig.add_subplot(grid[3])
    for j, policy in enumerate(POLICIES):
        color = _asset_baseline_policy_color(policy)
        for schedule in SCHEDULES:
            s = success[schedule]
            cell = s[(s.policy == policy) & (s.checkpoint == 20) & (s.budget == BUDGET)]
            assert len(cell) == 1
            ax.errorbar(j + OFFSET[schedule], cell.success.iloc[0], yerr=cell.sem_over_seeds.iloc[0], color=color,
                        lw=0.8, capsize=1.2, marker=CURVE_STYLE[policy]['marker'], ms=2.8, markeredgewidth=0.5,
                        **_fill(schedule, color))
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(rf'After 20 trials, {BUDGET * BUDGET_PA2_S:g} pA$^2$ s', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    _method_axis(ax, 'Task success'); _panel_label(ax, 'D', dx=-30)
    _legend(fig, plt)
    return save_figure(fig, output, plt_module=plt)


def _legend(fig, plt):
    gray = '#4A4A4A'
    handles = [plt.Line2D([], [], color=_asset_baseline_policy_color(p), lw=0, marker=CURVE_STYLE[p]['marker'],
                          ms=2.8, label=_asset_policy_label(p)) for p in POLICIES]
    handles += [plt.Line2D([], [], color=gray, ls=LINE['strong'], marker='o', ms=2.8, markerfacecolor='white',
                           label='20 pA throughout (selected)'),
                plt.Line2D([], [], color=gray, ls=LINE['weak'], marker='o', ms=2.8,
                           label='2 pA in evidence trials 1-10, then 20 pA')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=_ASSET_TICK_SIZE, columnspacing=0.9, handlelength=1.8, handletextpad=0.3, frameon=False)


def trials(folder: Path, output: Path) -> Path:
    """Evidence-led trials 1-10: identification input, flipped first choices against the evidence-only replay, decision time."""
    summary = pd.read_csv(folder / 'comparison' / 'evidence_summary.csv').set_index(['schedule', 'policy'])
    base = json.loads((folder / 'comparison' / 'evidence_only_summary.json').read_text())
    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 1.95))
    grid = fig.add_gridspec(1, 3, left=0.07, right=0.995, top=0.80, bottom=0.30, wspace=0.38)
    panels = [('control_rms_pa', 'RMS input (pA)', 'Identification input'),
              ('choice_flipped_pct', 'Decided trials (%)', 'Choice flipped vs evidence only'),
              ('median_decision_ms', 'Decision time (ms)', 'Decision time')]
    summary['choice_flipped_pct'] = 100 * summary.choice_flipped_vs_evidence_only / summary.decisions
    for i, (key, ylabel, title) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        for j, policy in enumerate(POLICIES):
            color = _asset_baseline_policy_color(policy)
            for schedule in SCHEDULES:
                ax.plot(j + OFFSET[schedule], summary.loc[(schedule, policy), key], lw=0, color=color,
                        marker=CURVE_STYLE[policy]['marker'], ms=3.0, markeredgewidth=0.6, **_fill(schedule, color))
        if key == 'control_rms_pa':
            ax.axhline(6.0, color='#8C8C8C', ls=':', lw=0.8)
            ax.text(len(POLICIES) - 0.45, 6.0, 'evidence 6 pA', ha='right', va='bottom', fontsize=_ASSET_TICK_SIZE,
                    color='#6E6861')
            ax.set_yscale('log'); ax.set_ylim(0.5, 40)
        elif key == 'median_decision_ms':
            ax.axhline(base['median_decision_ms'], color='#8C8C8C', ls=':', lw=0.8)
            ax.text(-0.5, base['median_decision_ms'] + 15, 'evidence only', ha='left', va='bottom',
                    fontsize=_ASSET_TICK_SIZE, color='#6E6861')
            ax.set_ylim(0, 900)
        else:
            ax.set_ylim(0, None)
        ax.set_title(title, fontsize=_ASSET_LABEL_SIZE, pad=2.0)
        _method_axis(ax, ylabel)
        _panel_label(ax, 'ABC'[i], dx=-30)
    _legend(fig, plt)
    return save_figure(fig, output, plt_module=plt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--experiment-dir', type=Path, required=True, help='weak experiment folder')
    parser.add_argument('--strong-dir', type=Path, default=FROZEN)
    parser.add_argument('--out-dir', type=Path, default=None, help='default: <experiment-dir>/figures')
    args = parser.parse_args(argv)
    out = args.out_dir or args.experiment_dir / 'figures'
    print(comparison({'strong': args.strong_dir, 'weak': args.experiment_dir}, out / 'spiking_weak_evidence_comparison.pdf'))
    print(trials(args.experiment_dir, out / 'spiking_weak_evidence_trials.pdf'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
