"""Plot verified control sweeps and paired PALDI comparisons at manuscript width.

Example:
  .venv/bin/python -m experiments.tnsre.figures.wong_wang_control \
    --models results/tnsre/20261005_wong_wang_control/inputs/models.json \
    --phase-dir results/tnsre/20261005_wong_wang_control/results/calibration_v1

Raw shards must have complete unique coverage. Sweep intervals are pointwise
crossed-bootstrap intervals of means. Final comparison intervals come directly
from the study's simultaneous paired analysis; these are distinct quantities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tnsre.wong_wang_control_report import paired_matrix, verified_rows
from .assets import _apply_asset_style, _asset_baseline_policy_color, save_figure
from .theme import style_experiment_axis

# Same 516-pt double-column canvas as spiking_control_budgets.py.
WIDTH = 516 / 72.27
ORDER = ['adaptive', 'active_myopic', 'flex_rollback', 'rhc', 'prbs', 'random',
         'prior', 'none', 'spread', 'front', 'pulse_half', 'pulse_tuned',
         'relay_default', 'relay_tuned', 'pi_default', 'pi_tuned', 'reduced_fit']
LABELS = dict(adaptive='PALDI', active_myopic='Myopic', flex_rollback='FLEX',
              rhc='RHC-US', prbs='PRBS', random='Random', prior='Initial model',
              none='No control', spread='Constant amplitude', front='Maximum amplitude',
              pulse_half='Half-amplitude pulse', pulse_tuned='Tuned pulse',
              relay_default='Relay', relay_tuned='Tuned relay', pi_default='PI',
              pi_tuned='Tuned PI', reduced_fit='Privileged fit')
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'x', '>', '<', 'h', '*', 'd', 'p', '+', 'H', '8']
INPUTS = ['uniform_current', 'uniform_poisson', 'partial_current', 'partial_poisson']
FAMILIES = dict(logistic='Sigmoid response model', transfer='Wong–Wang transfer model')


def arm_label(arm):
    family, target, evidence = arm.split('_')
    stimulation = 'uniform stimulation' if target == 'uniform' else 'partial, unequal stimulation'
    return f'{FAMILIES[family]} / {stimulation} / {evidence.capitalize()} evidence'


def ordered(policies):
    unknown = set(policies)-set(ORDER)
    if unknown:
        raise ValueError(f'Add explicit labels before plotting these policies: {unknown}')
    return [p for p in ORDER if p in set(policies)]


def color(policy, grayscale=False):
    c = _asset_baseline_policy_color(policy) if policy in ORDER[:6] else '#626262'
    if grayscale:
        from matplotlib.colors import to_rgb
        c = (sum(a*b for a, b in zip(to_rgb(c), [.2126, .7152, .0722])),)*3
    return c


def read_data(models, phase):
    """Recheck the saved summary against complete raw outcomes before rendering."""
    plan = json.loads((phase/'plan.json').read_text())
    data = pd.DataFrame(verified_rows(models, phase.parent, plan))
    if plan.get('purpose') == 'baseline_tuning':
        raise ValueError('Raw tuning candidates belong in their selection table')
    keys = ['arm', 'condition', 'policy']
    metrics = ['success', 'rmse', 'occupancy', 'energy']
    expected = (data[data.valid == 1].groupby([*keys, 'seed'])[metrics].mean()
                .groupby(keys).mean().sort_index())
    saved = pd.read_csv(phase/'comparison/control_summary.csv').set_index(keys).sort_index()
    if not expected.index.equals(saved.index):
        raise ValueError('Summary cells differ from the raw outcomes')
    np.testing.assert_allclose(expected[metrics], saved[metrics], rtol=1e-10, atol=1e-12,
                               equal_nan=True)
    metadata = data[['arm', 'condition', 'task', 'bound', 'budget']].drop_duplicates()
    summary = saved.reset_index().merge(metadata, on=['arm', 'condition'], validate='many_to_one')
    if (data.energy > data.budget+1e-6).any() or (data.peak_action_norm > data.bound+1e-6).any():
        raise ValueError('A recorded control exceeded its constraints')
    if data.groupby(['arm', 'task_seed']).valid.nunique().max() != 1:
        raise ValueError('Initial-decision exclusions differ across methods')
    if not data.success.isin([0, 1]).all():
        raise ValueError('Success must be a binary outcome')
    tracking = data[(data.valid == 1) & (data.task != 'overturn')]
    if not np.isfinite(tracking[['rmse', 'occupancy']]).all().all():
        raise ValueError('Tracking scores must be finite')
    return plan, data, summary


def sweep_intervals(data, plan, *, draws=8000, seed=20261005):
    """Pointwise 95% intervals, pairing model indices and network-noise seeds."""
    rng = np.random.default_rng(seed)
    train = plan['training_seeds']
    tw = rng.multinomial(len(train), np.full(len(train), 1/len(train)), size=draws)/len(train)
    seeds = plan['task_seeds']
    counts = rng.multinomial(len(seeds), np.full(len(seeds), 1/len(seeds)), size=draws)
    rows = []
    for (arm, condition), group in data[(data.valid == 1) & (data.task == 'overturn')].groupby(['arm', 'condition']):
        valid = sorted(group.task_seed.unique())
        weights = counts[:, [seeds.index(s) for s in valid]].astype(float)
        empty = weights.sum(1) == 0
        weights[empty] = 1.
        weights /= weights.sum(1, keepdims=True)
        for policy in ordered(group.policy.unique()):
            values = paired_matrix(group, policy, train, valid, 'success')
            samples = np.einsum('bi,ij,bj->b', tw, values, weights, optimize=True)
            lo, hi = np.quantile(samples, [.025, .975])
            rows.append(dict(arm=arm, condition=condition, policy=policy,
                mean=float(values.mean()), lower=lo, upper=hi,
                bound=float(group.bound.iloc[0]), budget=float(group.budget.iloc[0]),
                model_indices=len(train), valid_task_seeds=len(valid),
                degenerate=int(samples.std() <= 1e-12), draws=draws, seed=seed))
    return pd.DataFrame(rows)


def export(fig, plt, path, *, preview_only=False):
    """Preserve physical width and use the repository's canvas-clipping audit."""
    np.testing.assert_allclose(fig.get_size_inches()[0], WIDTH, rtol=0, atol=1e-10)
    fig.canvas.draw()
    for ax in fig.axes:
        for labels in [ax.get_xticklabels(), ax.get_yticklabels()]:
            visible = [t for t in labels if t.get_visible() and t.get_text()]
            if any(t.get_fontsize() < 6 for t in visible):
                raise ValueError('Text smaller than the manuscript 6-pt tick style')
            boxes = [t.get_window_extent(fig.canvas.get_renderer()) for t in visible]
            if any(a.overlaps(b) for a, b in zip(boxes, boxes[1:])):
                raise ValueError('Overlapping tick labels')
    # None alone inherits savefig.bbox='tight' from the manuscript rcParams.
    with plt.rc_context({'savefig.bbox': None}):
        # Append the extension: condition names contain decimal bounds.
        fig.savefig(path.parent/(path.name+'.png'), dpi=180, bbox_inches=None)
    if preview_only:
        plt.close(fig)
    else:
        save_figure(fig, path.parent/(path.name+'.pdf'), plt_module=plt)


def plot_sweeps(scores, out, plt, *, grayscale=False):
    """Separate energy and peak bounds instead of conflating the constraints."""
    for arm, group in scores.groupby('arm', sort=False):
        panels = [group[np.isclose(group.bound, 1.)], group[np.isclose(group.budget, 10.)]]
        if any(p[field].nunique() < 2 for p, field in zip(panels, ['budget', 'bound'])):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.95), sharey=True)
        fig.subplots_adjust(left=.085, right=.985, bottom=.31, top=.86, wspace=.24)
        fig.text(.5, .985, arm_label(arm), ha='center', va='top', fontsize=7.)
        for ax, panel, field, title in zip(axes, panels, ['budget', 'bound'],
                ['(a) Energy budget; peak bound 1', '(b) Peak bound; energy budget 10']):
            for policy in ordered(panel.policy.unique()):
                a = panel[panel.policy == policy].sort_values(field)
                idx = ORDER.index(policy)
                ax.plot(a[field], a['mean'], color=color(policy, grayscale),
                    marker=MARKERS[idx], markersize=3.2, linestyle=['-', '--', ':', '-.'][idx % 4],
                    linewidth=1.35 if policy == 'adaptive' else .85, label=LABELS[policy])
                ax.fill_between(a[field], a.lower, a.upper, color=color(policy, grayscale),
                                alpha=.10, linewidth=0)
            ax.set_ylim(-.015, 1.04)
            ax.set_yticks([0, .25, .5, .75, 1])
            ax.set_xticks(sorted(panel[field].unique()))
            ax.set_xlabel('Energy budget B (model units)' if field == 'budget'
                          else 'Input norm bound (dimensionless)')
            ax.set_title(title, loc='left', pad=6)
            style_experiment_axis(ax)
        axes[0].set_ylabel('Overturn success fraction')
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False,
                   columnspacing=1.2, handlelength=2., bbox_to_anchor=(.52, .01))
        suffix = '_grayscale' if grayscale else ''
        export(fig, plt, out/f'{arm}_constraints{suffix}', preview_only=grayscale)


def comparison_points(summary, comparisons):
    """Use simultaneous intervals, or explicitly descriptive single-seed points."""
    if len(comparisons):
        for row in comparisons.itertuples():
            group = summary[(summary.arm == row.arm) & (summary.condition == row.condition)]
            metric = 'success' if row.endpoint == 'success' else 'rmse'
            values = group.set_index('policy')[metric]
            delta = values['adaptive']-values[row.baseline]
            if metric == 'rmse':
                delta = -delta
            np.testing.assert_allclose(row.advantage, delta, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose([row.lower, row.upper],
                [row.advantage-row.simultaneous_critical*row.bootstrap_se,
                 row.advantage+row.simultaneous_critical*row.bootstrap_se], rtol=1e-10, atol=1e-12)
        return comparisons
    rows = []
    for (arm, condition), group in summary.groupby(['arm', 'condition']):
        if 'adaptive' not in set(group.policy):
            continue
        metric = 'success' if group.task.iloc[0] == 'overturn' else 'rmse'
        values = group.set_index('policy')[metric]
        for policy in ordered(set(group.policy)-{'adaptive'}):
            delta = (values['adaptive']-values[policy]) * (1 if metric == 'success' else -1)
            rows.append(dict(arm=arm, condition=condition, baseline=policy, endpoint=metric,
                advantage=delta, lower=np.nan, upper=np.nan, degenerate_bootstrap=1))
    return pd.DataFrame(rows)


def plot_comparisons(points, out, plt, *, grayscale=False):
    if points.empty:
        return
    for condition, all_arms in points.groupby('condition'):
        # One scale across both response families for this same endpoint.
        limits = all_arms[['advantage', 'lower', 'upper']].to_numpy().ravel()
        limits = limits[np.isfinite(limits)]
        lo, hi = min(0., limits.min()), max(0., limits.max())
        pad = max(.1, (hi-lo)*.08)
        factor = 100 if all_arms.endpoint.iloc[0] == 'success' else 1
        for family in ['logistic', 'transfer']:
            arms = [family+'_'+s for s in INPUTS if family+'_'+s in set(all_arms.arm)]
            if not arms:
                continue
            policies = ordered(all_arms.baseline.unique())
            nrows = (len(arms)+1)//2
            fig, axes = plt.subplots(nrows, 2, figsize=(WIDTH, 2.9*nrows+.35), squeeze=False)
            fig.subplots_adjust(left=.17, right=.985, bottom=.19/nrows, top=.90, wspace=.67, hspace=.45)
            header = FAMILIES[family]+': '+condition
            if not np.isfinite(all_arms.lower).any():
                header += '; descriptive means only, no confidence intervals'
            fig.text(.5, .99, header, ha='center', va='top', fontsize=7.)
            for index, (ax, arm) in enumerate(zip(axes.flat, arms)):
                group = all_arms[all_arms.arm == arm].set_index('baseline')
                if set(group.index) != set(policies):
                    raise ValueError('Missing baseline in a paired comparison panel')
                for y, policy in enumerate(policies):
                    row = group.loc[policy]
                    c = color(policy, grayscale)
                    if np.isfinite([row.lower, row.upper]).all():
                        ax.hlines(y, row.lower*factor, row.upper*factor, color=c, linewidth=1.)
                    ax.plot(row.advantage*factor, y, marker=MARKERS[ORDER.index(policy)],
                            color=c, markerfacecolor='white' if row.degenerate_bootstrap else c,
                            markeredgecolor='#444444', markeredgewidth=.65,
                            markersize=3.6, linestyle='none')
                ax.axvline(0, color=color('adaptive', grayscale), linewidth=.8, linestyle='--', zorder=0)
                ax.set_yticks(range(len(policies)), [LABELS[p] for p in policies])
                ax.set_ylim(len(policies)-.4, -.6)
                ax.set_xlim((lo-pad)*factor, (hi+pad)*factor)
                ax.set_title(f'({chr(97+index)}) '+arm.split('_', 1)[1].replace('_', ' / '), loc='left', pad=6)
                ax.set_xlabel('PALDI − baseline success (pp)' if factor == 100
                              else 'Baseline − PALDI RMSE (proxy units)')
                style_experiment_axis(ax)
                ax.xaxis.grid(True, color='#DDDDDD', linewidth=.4)
            for ax in list(axes.flat)[len(arms):]:
                ax.set_axis_off()
            suffix = '_grayscale' if grayscale else ''
            export(fig, plt, out/f'{family}_{condition}_paired{suffix}', preview_only=grayscale)


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _apply_asset_style(plt)
    # Use the manuscript stack's bundled TrueType fallback; the local
    # Helvetica substitute is CFF and triggers PDF font-type warnings.
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    return plt


def render_diagnostics(models, phase):
    """Show all saved PALDI candidate replays; these are explanatory old seeds."""
    plan = json.loads((phase/'plan.json').read_text())
    rows = verified_rows(models, phase.parent, plan)
    records = []
    for r in rows:
        if r['policy'] != 'adaptive' or not r['valid']:
            continue
        for j, candidate in enumerate(r['candidates']):
            records.append(dict(arm=r['arm'], seed=r['seed'], task_seed=r['task_seed'],
                candidate=candidate, filtered=r['prediction']['filtered']['hard'][j],
                true_onset=r['prediction']['true_onset']['hard'][j], actual=r['actual'][j]['success']))
    data = pd.DataFrame(records)
    out = phase/'comparison/figures'
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out/'diagnostic_points.csv', index=False)
    arms = [a for a in plan['arms'] if a in set(data.arm)]
    plt = plotting()
    for gray in [False, True]:
        nrows = (len(arms)+1)//2
        fig, axes = plt.subplots(nrows, 2, figsize=(WIDTH, 2.7*nrows), squeeze=False, sharey=True)
        fig.subplots_adjust(left=.08, right=.985, bottom=.24/nrows, top=.86, wspace=.22, hspace=.4)
        for index, (ax, arm) in enumerate(zip(axes.flat, arms)):
            group = data[data.arm == arm]
            candidates = ['selected', 'front', 'spread']
            means = group.groupby('candidate')[['filtered', 'true_onset', 'actual']].mean().loc[candidates]
            for j, (key, label, shade, hatch) in enumerate([
                    ('filtered', 'Model: filtered onset', '.82', '//'),
                    ('true_onset', 'Model: true onset (privileged)', '.62', '..'),
                    ('actual', 'SNN replay', '.35', '')]):
                xs = np.arange(3)+(j-1)*.24
                ax.bar(xs, means[key], width=.21, facecolor=shade, edgecolor='#333333',
                       linewidth=.4, hatch=hatch, label=label, zorder=3)
                if key == 'actual':
                    for x, candidate in zip(xs, candidates):
                        a = group[group.candidate == candidate].actual
                        ax.text(x, a.mean()+.025, f'{int(a.sum())}/{len(a)}', ha='center', fontsize=6.)
            ax.set_xticks(range(3), ['Selected plan', 'Maximum pulse', 'Constant pulse'])
            ax.set_ylim(0, 1.13)
            ax.set_yticks([0, .25, .5, .75, 1.])
            ax.set_ylabel('Predicted / observed success')
            ax.set_title(f'({chr(97+index)}) '+arm.replace('_', ' / '), loc='left')
            style_experiment_axis(ax)
        for ax in list(axes.flat)[len(arms):]:
            ax.set_axis_off()
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                   bbox_to_anchor=(.51, .02))
        fig.text(.5, .985, 'PALDI: full open-loop plan replay; previously used diagnostic seeds',
                 ha='center', va='top', fontsize=7.)
        suffix = '_grayscale' if gray else ''
        export(fig, plt, out/f'paldi_open_loop_replay{suffix}', preview_only=gray)
    inputs = [models, phase/'plan.json', Path(__file__), *sorted(phase.glob('*/diagnostic_*.jsonl'))]
    (out/'provenance.json').write_text(json.dumps(dict(phase=plan['name'],
        verified_rows=len(rows), plotted_policy='adaptive', width_inches=WIDTH,
        model_indices=sorted(data.seed.unique().tolist()),
        task_seeds=sorted(data.task_seed.unique().tolist()),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}), indent=2)+'\n')
    means = data.groupby(['arm', 'candidate'])[['filtered', 'actual']].mean()
    reversed_rank = sum(means.loc[(arm, 'selected'), 'filtered'] > means.loc[(arm, 'front'), 'filtered']
        and means.loc[(arm, 'selected'), 'actual'] < means.loc[(arm, 'front'), 'actual'] for arm in arms)
    (out/'CAPTION.txt').write_text(
        f"The predicted selected-versus-maximum-pulse ranking reverses in {reversed_rank}/{len(arms)} "
        "input conditions when replayed in the full SNN. "
        "Every saved PALDI candidate is shown; the complete diagnostic also retains the other methods. "
        "Bars give mean predicted hard first-passage probabilities from the filtered onset state, "
        "from the true onset state (privileged diagnostic), and observed SNN success. "
        "Counts above the SNN bars show successful replays divided by all valid replays. "
        "Each candidate uses the same uncontrolled prefix and then runs its complete open-loop plan; "
        "these are not closed-loop MPC success rates. The model index and reused task seeds are "
        "listed in provenance.json. This small explanatory panel is not fresh confirmation; "
        "it has no inferential uncertainty marks. No smoothing or normalization is applied.\n")
    print(json.dumps(dict(phase=plan['name'], verified_rows=len(rows), output=str(out))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--phase-dir', type=Path, required=True)
    args = parser.parse_args()
    if json.loads((args.phase_dir/'plan.json').read_text()).get('worker_stage') == 'diagnostic':
        render_diagnostics(args.models, args.phase_dir)
        return
    plan, data, summary = read_data(args.models, args.phase_dir)
    out = args.phase_dir/'comparison/figures'
    out.mkdir(parents=True, exist_ok=True)
    sweeps = sweep_intervals(data, plan)
    sweeps.to_csv(out/'constraint_points.csv', index=False)
    path = args.phase_dir/'comparison/paired_comparisons.csv'
    try:
        comparisons = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        comparisons = pd.DataFrame()
    points = comparison_points(summary, comparisons)
    points.to_csv(out/'paired_points.csv', index=False)
    plt = plotting()
    for gray in [False, True]:
        if len(sweeps):
            plot_sweeps(sweeps, out, plt, grayscale=gray)
        plot_comparisons(points, out, plt, grayscale=gray)
    inputs = [args.models, args.phase_dir/'plan.json', Path(__file__), args.phase_dir/'comparison/control_summary.csv',
              path, *sorted(args.phase_dir.glob('*/worker_*.jsonl'))]
    provenance = dict(phase=plan['name'], width_inches=WIDTH, font='DejaVu Sans',
        model_indices=plan['training_seeds'], task_seeds=plan['task_seeds'],
        numerical_failures=int(data.numerical_failure.sum()),
        invalid_trials=int((data.valid == 0).sum()),
        constraints='Pointwise 95% crossed-bootstrap percentile intervals; 8000 draws, seed 20261005',
        paired='Saved simultaneous 95% paired crossed-bootstrap intervals; positive favors PALDI. If absent, descriptive means with no intervals.',
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    caption=(f"Phase: {plan['name']}. {len(plan['training_seeds'])} frozen model/readout indices and "
        f"{len(plan['task_seeds'])} planned network-noise seeds. Invalid initial decisions are excluded "
        f"identically across controllers within each input condition ({provenance['invalid_trials']} trial rows); "
        f"{provenance['numerical_failures']} numerical failures remain in the scores. "
        "Constraint panels vary total energy at peak bound 1 and peak bound at energy budget 10 separately. "
        "B is the sum of squared input norms times the model time step (100-ms time units). "
        "Curves show mean first-passage success; shading shows pointwise 95% crossed-bootstrap intervals "
        "over model/readout indices and network-noise seeds. Fixed pulses are broadcast across model indices, "
        "not treated as independent repeat trials. Degenerate all-zero or all-one samples give collapsed "
        "bootstrap intervals and do not establish population probabilities of zero or one. "
        "These intervals of individual means are not tests of differences. Paired panels show baseline minus "
        "PALDI tracking RMSE, or PALDI minus baseline success in percentage points; positive favors PALDI. "
        "Their intervals use the saved simultaneous 95% paired analysis across all reported comparisons. "
        "Hollow marks identify unavailable or degenerate intervals and cannot establish superiority. A single model "
        "seed cannot establish an advantage across learned models. No smoothing or normalization is applied. "
        "Panels within a condition share limits across input arms and response families; "
        "policy order is fixed, not ranked.\n")
    (out/'CAPTION.txt').write_text(caption)
    captions = {}
    if len(sweeps):
        for arm, group in sweeps.groupby('arm'):
            low = group[group.bound == group.bound.min()]
            captions[arm+'_constraints'] = (
                f"{arm_label(arm)}. At the smallest tested bound ({group.bound.min():g}), "
                f"{int((low['mean'] > 0).sum())}/{len(low)} controller means are above zero. "+caption)
    if len(points):
        for condition, group in points.groupby('condition'):
            for family in ['logistic', 'transfer']:
                panel = group[group.arm.str.startswith(family+'_')]
                if panel.empty:
                    continue
                positive = int((panel.advantage > 0).sum())
                established = int(((panel.lower > 0) & (panel.degenerate_bootstrap == 0)).sum())
                captions[f'{family}_{condition}_paired'] = (
                    f"{FAMILIES[family]}, condition {condition}. "
                    f"Mean differences favor PALDI in {positive}/{len(panel)} comparisons; "
                    f"{established} have nondegenerate simultaneous intervals entirely above zero. "+caption)
    (out/'captions.json').write_text(json.dumps(captions, indent=2)+'\n')
    print(json.dumps(dict(phase=plan['name'],verified_rows=len(data),output=str(out),
                         vector_figures=len(list(out.glob('*.pdf'))))))


if __name__ == '__main__':
    main()
