"""Figures comparing the SNN identification scenarios (no simulation).

Inputs: the scenario folder written by ``experiments.tnsre.snn_evidence_scenarios``, the 2-pA run
(``results/archive/experiments/20261007_snn_weak_evidence``), and the selected result
(``results/archive/experiments/20261007_snn_native_uniform``). All conditions share the 20 training seeds,
readouts, priors, network seeds, and task panel.

* ``spiking_evidence_scenarios``: (A) PALDI held-out R2 along identification (median over seeds),
  (B) R2 after 20 trials for every method (median, IQR), (C) PALDI overturn success at 480 pA^2 s
  along identification, (D) success after 20 trials (mean +/- SEM over seeds). The passive
  learner (20 evidence trials, no control) is the gray line in A and C and the gray band in B and D.
* ``spiking_evidence_scenarios_trials``: evidence-led trials of each condition: (A) RMS
  identification input, (B) first choices flipped relative to the evidence-only replay with the
  same network seed, (C) median decision time, (D) total identification time of 20 trials.

Usage:
    python -m experiments.tnsre.figures.spiking_evidence_scenarios --experiment-dir <scenario folder>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from actdyn.utils.figure_io import load_plotting

from .assets import _ASSET_LABEL_SIZE, _ASSET_TICK_SIZE, _apply_asset_style, _asset_policy_label, save_figure
from .spiking_sessions import BUDGET_PA2_S, CONTROL_CHECKPOINTS, FIGURE_WIDTH, POLICIES, _panel_label
from .theme import style_experiment_axis

FROZEN = Path('results/archive/experiments/20261007_snn_native_uniform')
WEAK2 = Path('results/archive/experiments/20261007_snn_weak_evidence')
ACTIVE = ('strong', 'weak2_e10', 'weak10_e10', 'weak2_e5')
PASSIVE, PASSIVE_POLICY = 'passive_e20', 'off_policy'
# Condition encoding shared by every panel (Okabe-Ito colors; line style and marker also differ).
STYLE = {
    'strong': dict(color='#333333', ls='-', marker='o', label='20 pA in evidence trials 1-10 (selected)'),
    'weak2_e10': dict(color='#0072B2', ls='--', marker='s', label='2 pA in evidence trials 1-10'),
    'weak10_e10': dict(color='#E69F00', ls='-.', marker='^', label='10 pA in evidence trials 1-10'),
    'weak2_e5': dict(color='#009E73', ls=(0, (1.2, 1.2)), marker='D', label='2 pA in evidence trials 1-5'),
    'passive_e20': dict(color='#8C8C8C', ls=':', marker='x', label='No control, 20 evidence trials'),
}
OFFSET = dict(zip(ACTIVE, (-0.27, -0.09, 0.09, 0.27)))
BUDGET = 12.0  # normalized; 480 pA^2 s
R2_LIM = (-0.5, 1.0)  # A and B; medians after trial 1 reach -0.43 (2 pA, passive)
R2_TICKS = (-0.5, 0.0, 0.5, 1.0)


def _folders(experiment_dir: Path, strong_dir: Path, weak2_dir: Path) -> dict[str, Path]:
    return {'strong': strong_dir, 'weak2_e10': weak2_dir,
            **{c: experiment_dir / 'scenarios' / c for c in ('weak10_e10', 'weak2_e5', PASSIVE)}}


def _r2(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / 'eval' / 'r2_per_session.csv')


def _success(folder: Path) -> pd.DataFrame:
    summary = pd.read_csv(folder / 'eval' / 'task_summary_warm.csv')
    return summary[summary.task == 'overturn']


def _line(ax, condition: str, x, y, **kwargs):
    s = STYLE[condition]
    ax.plot(x, y, color=s['color'], ls=s['ls'], marker=s['marker'], lw=0.95, ms=2.4, markeredgewidth=0.5, **kwargs)


def _point(ax, condition: str, x, y, yerr):
    s = STYLE[condition]
    ax.errorbar(x, y, yerr=yerr, color=s['color'], lw=0.8, capsize=1.2, marker=s['marker'], ms=2.6,
                markeredgewidth=0.5, ls='none')


def _passive_band(ax, lo: float, mid: float, hi: float):
    """The passive learner as a reference band across all methods."""
    s = STYLE[PASSIVE]
    ax.axhspan(lo, hi, color=s['color'], alpha=0.16, lw=0, zorder=0)
    ax.axhline(mid, color=s['color'], ls=s['ls'], lw=0.9, zorder=0.5)


def _method_axis(ax, ylabel: str):
    ax.set_xticks(range(len(POLICIES)), [_asset_policy_label(p) for p in POLICIES], rotation=30, ha='right',
                  rotation_mode='anchor')
    ax.set_xlim(-0.6, len(POLICIES) - 0.4)
    ax.set_ylabel(ylabel)
    style_experiment_axis(ax)


def _legend(fig, plt, conditions):
    handles = [plt.Line2D([], [], color=STYLE[c]['color'], ls=STYLE[c]['ls'], marker=STYLE[c]['marker'], ms=2.6,
                          lw=0.95, label=STYLE[c]['label']) for c in conditions]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=_ASSET_TICK_SIZE, columnspacing=1.0, handlelength=2.2, handletextpad=0.35, frameon=False)


def comparison(folders: dict[str, Path], output: Path) -> Path:
    r2 = {c: _r2(p) for c, p in folders.items()}
    success = {c: _success(p) for c, p in folders.items()}
    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.2))
    grid = fig.add_gridspec(1, 4, left=0.065, right=0.995, top=0.80, bottom=0.24, wspace=0.42,
                            width_ratios=[1.0, 1.1, 1.0, 1.1])
    policy_of = lambda c: PASSIVE_POLICY if c == PASSIVE else 'adaptive'
    # A: PALDI (and the passive learner) R2 along identification; R2 after trial k sits at x = k.
    ax = fig.add_subplot(grid[0])
    for c in (*ACTIVE, PASSIVE):
        q = r2[c][r2[c].policy == policy_of(c)].groupby('sessions').r2.median()
        assert (q.index == np.arange(21)).all()
        _line(ax, c, q.index, q.to_numpy(), markevery=5)
    ax.set_xlim(0, 20); ax.set_ylim(*R2_LIM); ax.set_yticks(R2_TICKS); ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel('Identification trials'); ax.set_ylabel(r'Held-out $R^2_{\mathrm{roll}}$ (median)')
    ax.set_title('PALDI prediction', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    style_experiment_axis(ax); _panel_label(ax, 'A', dx=-30)
    # B: R2 after 20 trials, median and IQR over seeds; passive learner as the band.
    ax = fig.add_subplot(grid[1])
    v = r2[PASSIVE][(r2[PASSIVE].policy == PASSIVE_POLICY) & (r2[PASSIVE].sessions == 20)].r2.to_numpy()
    assert len(v) == 20
    _passive_band(ax, *np.percentile(v, [25, 50, 75]))
    for j, policy in enumerate(POLICIES):
        for c in ACTIVE:
            v = r2[c][(r2[c].policy == policy) & (r2[c].sessions == 20)].r2.to_numpy()
            assert len(v) == 20
            lo, med, hi = np.percentile(v, [25, 50, 75])
            _point(ax, c, j + OFFSET[c], med, [[med - lo], [hi - med]])
    ax.set_ylim(*R2_LIM); ax.set_yticks(R2_TICKS)
    ax.set_title('After 20 trials', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    _method_axis(ax, r'Held-out $R^2_{\mathrm{roll}}$'); _panel_label(ax, 'B', dx=-30)
    # C: PALDI success at 480 pA^2 s along identification (checkpoint 0 = the shared prior).
    ax = fig.add_subplot(grid[2])
    pos = np.arange(len(CONTROL_CHECKPOINTS))
    for c in (*ACTIVE, PASSIVE):
        s = success[c]
        pts = [s[(s.policy == ('prior' if k == 0 else policy_of(c))) & (s.checkpoint == k) & (s.budget == BUDGET)]
               for k in CONTROL_CHECKPOINTS]
        assert all(len(p) == 1 for p in pts), c
        _line(ax, c, pos, [float(p.success.iloc[0]) for p in pts])
    ax.set_xticks(pos, [str(k) for k in CONTROL_CHECKPOINTS]); ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Identification trials'); ax.set_ylabel('Task success (mean)')
    ax.set_title(rf'PALDI overturn, {BUDGET * BUDGET_PA2_S:g} pA$^2$ s', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    style_experiment_axis(ax); _panel_label(ax, 'C', dx=-30)
    # D: success after 20 trials, mean +/- SEM over training seeds; passive learner as the band.
    ax = fig.add_subplot(grid[3])
    s = success[PASSIVE]
    cell = s[(s.policy == PASSIVE_POLICY) & (s.checkpoint == 20) & (s.budget == BUDGET)]
    assert len(cell) == 1
    m, e = float(cell.success.iloc[0]), float(cell.sem_over_seeds.iloc[0])
    _passive_band(ax, m - e, m, m + e)
    for j, policy in enumerate(POLICIES):
        for c in ACTIVE:
            s = success[c]
            cell = s[(s.policy == policy) & (s.checkpoint == 20) & (s.budget == BUDGET)]
            assert len(cell) == 1
            _point(ax, c, j + OFFSET[c], float(cell.success.iloc[0]), float(cell.sem_over_seeds.iloc[0]))
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(rf'After 20 trials, {BUDGET * BUDGET_PA2_S:g} pA$^2$ s', fontsize=_ASSET_LABEL_SIZE, pad=2.0)
    _method_axis(ax, 'Task success'); _panel_label(ax, 'D', dx=-30)
    _legend(fig, plt, (*ACTIVE, PASSIVE))
    return save_figure(fig, output, plt_module=plt)


def trials(experiment_dir: Path, output: Path) -> Path:
    """Evidence-led trials of every condition and the total identification time."""
    summary = pd.read_csv(experiment_dir / 'comparison' / 'evidence_summary.csv').set_index(['condition', 'policy'])
    summary['choice_flipped_pct'] = 100 * summary.choice_flipped_vs_evidence_only / summary.decisions
    passive = summary.loc[(PASSIVE, PASSIVE_POLICY)]
    plt = load_plotting(output, apply_style=_apply_asset_style, path_is_file=True)
    fig = plt.figure(figsize=(FIGURE_WIDTH, 2.05))
    grid = fig.add_gridspec(1, 4, left=0.06, right=0.995, top=0.80, bottom=0.30, wspace=0.42)
    panels = [('control_rms_pa', 'RMS input (pA)', 'Control input'),
              ('choice_flipped_pct', 'Decided trials (%)', 'Flipped vs evidence only'),
              ('median_decision_ms', 'Median (ms)', 'Decision time'),
              ('identification_s', 'Time (s)', 'All 20 trials')]
    for i, (key, ylabel, title) in enumerate(panels):
        ax = fig.add_subplot(grid[i])
        if key in ('median_decision_ms', 'identification_s'):
            _passive_band(ax, passive[key], passive[key], passive[key])
        for j, policy in enumerate(POLICIES):
            for c in ACTIVE:
                s = STYLE[c]
                ax.plot(j + OFFSET[c], summary.loc[(c, policy), key], lw=0, color=s['color'], marker=s['marker'],
                        ms=2.8, markeredgewidth=0.5)
        if key == 'control_rms_pa':
            ax.axhline(6.0, color='#8C8C8C', ls=':', lw=0.8)
            ax.text(-0.5, 5.6, 'evidence 6 pA', ha='left', va='top', fontsize=_ASSET_TICK_SIZE,
                    color='#6E6861')
            ax.set_yscale('log'); ax.set_ylim(0.5, 40)
        elif key == 'median_decision_ms':
            ax.set_ylim(0, 1000)
        else:
            ax.set_ylim(0, None)
        ax.set_title(title, fontsize=_ASSET_LABEL_SIZE, pad=2.0)
        _method_axis(ax, ylabel)
        _panel_label(ax, 'ABCD'[i], dx=-30)
    _legend(fig, plt, (*ACTIVE, PASSIVE))
    return save_figure(fig, output, plt_module=plt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--experiment-dir', type=Path, required=True, help='scenario comparison folder')
    parser.add_argument('--strong-dir', type=Path, default=FROZEN)
    parser.add_argument('--weak2-dir', type=Path, default=WEAK2)
    parser.add_argument('--out-dir', type=Path, default=None, help='default: <experiment-dir>/figures')
    args = parser.parse_args(argv)
    out = args.out_dir or args.experiment_dir / 'figures'
    folders = _folders(args.experiment_dir, args.strong_dir, args.weak2_dir)
    print(comparison(folders, out / 'spiking_evidence_scenarios.pdf'))
    print(trials(args.experiment_dir, out / 'spiking_evidence_scenarios_trials.pdf'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
