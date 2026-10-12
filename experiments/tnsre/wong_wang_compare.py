"""Validate the complete manuscript ablation and make tables and paired plots.

Intervals resample training seeds, conditional on the fixed eight task seeds.
No task trial is counted as an independent trained model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tnsre.wong_wang_full import TRAIN_ARMS, EVAL_ARMS, MODEL_IDS, configure_eval
from experiments.tnsre import eval_spiking_sessions as ev

LABELS = {'adaptive': 'PALDI', 'active_myopic': 'Myopic', 'flex_rollback': 'FLEX',
          'rhc': 'RHC', 'prbs': 'PRBS', 'random': 'Random'}


def paired_interval(x, y):
    """Mean paired difference and percentile 95% interval, 10,000 seed resamples."""
    d = np.asarray(x) - np.asarray(y)
    rng = np.random.default_rng(20261004)
    boot = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return float(d.mean()), *np.quantile(boot, [.025, .975]).tolist()


def read_and_verify(root):
    r2s, tasks, sources = [], [], {}
    for arm in TRAIN_ARMS:
        paths = list((root / 'tracks' / f'ww_full_{arm}').glob('*/seed_*/repeat_01/run_metadata.json'))
        expected = {(policy, seed) for policy in MODEL_IDS for seed in range(20)}
        actual = set()
        for path in paths:
            d = json.loads(path.read_text())
            assert d['status'] == 'completed' and not d.get('nan_detected'), path
            assert len(d['session_log']) == 20 and np.isfinite(d['embedding_estimate']).all(), path
            actual.add((d['policy_id'], int(d['seed'])))
        assert len(paths) == 120 and actual == expected, arm
        path = root / 'eval' / f'{arm}_current' / 'r2_per_session.csv'
        df = pd.read_csv(path)
        assert len(df) == 2520 and not df.duplicated(['policy', 'seed', 'sessions']).any()
        assert np.isfinite(df.r2).all()
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        r2s.append(df.assign(arm=arm))
    for arm in EVAL_ARMS:
        runs_root = configure_eval(root, arm)
        units = ev.controller_units(ev.load_session_runs(runs_root))
        expected = {ev._job_key(units[i], task, budget, seed) for i, task, budget, seed in ev.task_jobs(units)}
        dfs = []
        for path in sorted((root / 'eval' / arm / 'task_warm').glob('worker_*.csv')):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            dfs.append(pd.read_csv(path, keep_default_na=False))
        df = pd.concat(dfs, ignore_index=True)
        keycols = ['controller', 'policy', 'seed', 'checkpoint', 'task', 'budget', 'task_seed']
        actual = set(df[keycols].itertuples(index=False, name=None))
        assert len(df) == 8360 and len(actual) == len(df) and actual == expected, arm
        assert np.isfinite(df.energy).all() and (df.energy <= df.budget + 1e-6).all()
        assert df.groupby('task_seed').valid.nunique().max() == 1, 'Unpaired validity within condition'
        tasks.append(df.assign(arm=arm))
    return pd.concat(r2s, ignore_index=True), pd.concat(tasks, ignore_index=True), sources


def plot_differences(table, path, xlabel, groups, caption):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from experiments.tnsre.figures.theme import apply_style, style_axis
    from experiments.tnsre.figures.assets import save_figure
    apply_style(plt)
    fig, axes = plt.subplots(1, len(groups), figsize=(7.1, 2.8), sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=.12, right=.98, top=.84, bottom=.24, wspace=.12)
    for ax, group in zip(axes, groups):
        df = table[table.condition == group].set_index('policy').loc[MODEL_IDS]
        y = np.arange(len(MODEL_IDS))
        ax.errorbar(df.delta, y, xerr=np.stack([np.maximum(0, df.delta-df.lo), np.maximum(0, df.hi-df.delta)]),
                    fmt='o', color='#3465a4', ecolor='.35', markersize=4, capsize=2)
        ax.axvline(0, ls='--', lw=.7, color='.5')
        ax.set_yticks(y, [LABELS[k] for k in MODEL_IDS])
        ax.text(.5, 1.06, group.replace('_', '\n'), transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
        style_axis(ax, grid_axis='x')
    axes[0].invert_yaxis()
    fig.supxlabel(xlabel, y=.02, fontsize=9)
    save_figure(fig, path.with_suffix('.pdf'), plt_module=plt)
    with plt.rc_context({'savefig.bbox': None}):
        fig.savefig(path.with_suffix('.png'), dpi=180, bbox_inches=None)
    plt.close(fig)
    path.with_suffix('.caption.txt').write_text(caption + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    r2, task, sources = read_and_verify(root)
    out = root / 'comparison'
    out.mkdir(exist_ok=True)
    r2.to_csv(out/'r2_all.csv', index=False)
    valid = task[task.valid == 1]
    numerical_failures = task[task.get('numerical_failure', pd.Series(0, index=task.index)).fillna(0).astype(int) == 1]
    numerical_failures.to_csv(out/'numerical_failures.csv', index=False)
    seed_scores = valid.groupby(['arm', 'policy', 'seed', 'checkpoint', 'budget'], as_index=False).agg(
        success=('success', 'mean'), energy=('energy', 'mean'), n_tasks=('success', 'size'))
    seed_scores.to_csv(out/'control_by_seed.csv', index=False)
    final = r2[r2.sessions == 20]
    r2_summary = final.groupby(['arm', 'policy']).r2.agg(median='median', q25=lambda v:v.quantile(.25),
                  q75=lambda v:v.quantile(.75), mean='mean', n='size', below_zero=lambda v:(v<0).sum()).reset_index()
    r2_summary.to_csv(out/'prediction_summary.csv', index=False)
    control_summary = seed_scores.groupby(['arm', 'policy', 'checkpoint', 'budget']).success.agg(
        mean='mean', median='median', q25=lambda v:v.quantile(.25), q75=lambda v:v.quantile(.75), n='size').reset_index()
    control_summary.to_csv(out/'control_summary.csv', index=False)
    reference_policies = ['prior', 'reduced_fit', 'none', 'spread', 'front']
    reference_summary = control_summary[control_summary.policy.isin(reference_policies)]
    reference_summary.to_csv(out/'reference_control_summary.csv', index=False)
    differences = []
    for target in ('uniform', 'partial'):
        for policy in MODEL_IDS:
            values = [final[(final.arm == f'{family}_{target}') & (final.policy == policy)].sort_values('seed').r2
                      for family in ('transfer', 'logistic')]
            delta, lo, hi = paired_interval(*values)
            differences.append(dict(condition=target, policy=policy, delta=delta, lo=lo, hi=hi))
    rd = pd.DataFrame(differences)
    rd.to_csv(out/'prediction_response_differences.csv', index=False)
    differences = []
    for target in ('uniform', 'partial'):
        for mode in ('current', 'poisson'):
            for policy in MODEL_IDS:
                values = [seed_scores[(seed_scores.arm == f'{family}_{target}_{mode}') &
                    (seed_scores.policy == policy) & (seed_scores.checkpoint == 20) &
                    (seed_scores.budget == 12.)].sort_values('seed').success for family in ('transfer', 'logistic')]
                delta, lo, hi = paired_interval(*values)
                differences.append(dict(condition=f'{target}_{mode}', policy=policy, delta=delta, lo=lo, hi=hi))
    cd = pd.DataFrame(differences)
    cd.to_csv(out/'control_response_differences.csv', index=False)
    # Input changes: paired training seeds and only task seeds valid in BOTH conditions.
    input_diffs = []
    for family in ('logistic', 'transfer'):
        base_arm = f'{family}_uniform_current'
        for suffix in ('partial_current', 'uniform_poisson', 'partial_poisson'):
            arm = f'{family}_{suffix}'
            for policy in MODEL_IDS:
                filt = (task.policy == policy) & (task.checkpoint == 20) & (task.budget == 12.)
                x, y = [task[filt & (task.arm == name) & (task.valid == 1)] for name in (arm, base_arm)]
                paired = x.merge(y, on=['seed','task_seed'], suffixes=('_x','_y'), validate='one_to_one')
                by_seed = paired.groupby('seed')[['success_x','success_y']].mean().sort_index()
                assert len(by_seed) == 20
                delta, lo, hi = paired_interval(by_seed.success_x, by_seed.success_y)
                input_diffs.append(dict(family=family, condition=suffix, policy=policy, delta=delta, lo=lo, hi=hi,
                                        common_task_seeds=sorted(paired.task_seed.unique().tolist())))
    input_diffs = pd.DataFrame(input_diffs)
    input_diffs.to_csv(out/'input_differences_common_tasks.csv', index=False)
    validity = task.groupby(['arm','task_seed']).valid.first().unstack()
    validity.to_csv(out/'task_validity.csv')
    caption = ('Transfer minus logistic after 20 identification trials. Dots show the mean paired difference; '
               'bars show a percentile 95% bootstrap interval over 20 matched training seeds (10,000 resamples). '
               'All six policies are shown. No smoothing or excluded training runs. Each family has its configured '
               'prior; these comparisons include that prior choice.')
    plot_differences(rd, out/'prediction_response', 'Difference in held-out rollout $R^2$', ['uniform','partial'], caption)
    plot_differences(cd, out/'control_response', 'Difference in change-of-mind success fraction',
                     ['uniform_current','uniform_poisson','partial_current','partial_poisson'], caption +
                     ' Control uses budget 12 (nominal 480 pA² s); intervals are conditional on the fixed task seeds. '
                     'No-initial-decision sessions are excluded within each condition; see task_validity.csv.')
    historical = root / 'historical'
    history_rows = []
    history_control = []
    if (historical/'r2_per_session.csv').exists():
        old = pd.read_csv(historical/'r2_per_session.csv')
        for policy in MODEL_IDS:
            old_values = old[(old.policy == policy) & (old.sessions == 20)].sort_values('seed').r2
            new_values = final[(final.arm == 'logistic_uniform') & (final.policy == policy)].sort_values('seed').r2
            history_rows.append(dict(policy=policy, old_median=float(old_values.median()),
                 new_median=float(new_values.median()), mean_paired_delta=float(np.mean(new_values.to_numpy()-old_values.to_numpy()))))
        pd.DataFrame(history_rows).to_csv(out/'historical_prediction_comparison.csv', index=False)
    if (historical/'task_summary_warm.csv').exists():
        old = pd.read_csv(historical/'task_summary_warm.csv')
        old = old[(old.task == 'overturn') & (old.checkpoint == 20) & (old.budget == 12.)]
        for policy in MODEL_IDS:
            old_row = old[old.policy == policy].iloc[0]
            new_row = control_summary[(control_summary.arm == 'logistic_uniform_current') &
                (control_summary.policy == policy) & (control_summary.checkpoint == 20) &
                (control_summary.budget == 12.)].iloc[0]
            history_control.append(dict(policy=policy, old_success=float(old_row.success),
                new_success=float(new_row['mean']), delta=float(new_row['mean']-old_row.success),
                old_n_seeds=int(old_row.n_seeds), new_n_seeds=int(new_row.n)))
        pd.DataFrame(history_control).to_csv(out/'historical_control_comparison.csv', index=False)
    lines = ['# Full Wong–Wang manuscript comparison', '',
             'Completed: 480 identification runs, 20 trials each, six policies and 20 matched seeds; '
             '66,880 change-of-mind task evaluations across eight conditions.', '',
             f'{len(numerical_failures)} evaluations ended with a numerical controller failure. '
             'Their outcomes use the recorded SNN prefix; a crash before the goal is reached counts as unsuccessful. '
             'These cases remain in valid-task denominators. Original shard failures, traces, and recovery logs are retained.', '',
             'Training has no sensory evidence. Prediction uses 12 held-out SNN trajectories, '
             '100-bin rollouts, stride 20, and pooled target variance. Control uses checkpoints 0, 1, 2, 5, 20; '
             'eight task seeds; budgets 10 and 12; and the manuscript warm-start iCEM settings.', '',
             '## Main findings', '',
             'PALDI works with the original Wong–Wang response. Removing the sigmoid keeps its prediction '
             'and control performance close to the fresh sigmoid baseline. The paired control intervals '
             'include zero in all four input conditions; this does not establish equivalence. '
             'The result does not extend to every policy: Myopic loses substantial prediction and control performance.', '',
             'Partial, unequal stimulation increases PALDI control success for both response functions. '
             'Total pool current is matched, but the squared-current dose for a given action is about 2.13 times '
             'larger and peak neuron current is higher. This is not an equal-dose improvement.', '',
             'Replacing uniform sensory current with synaptic Poisson evidence preserves useful PALDI control. '
             'On the common valid task seeds, the small decrease in success has a paired interval that includes '
             'zero for both response functions. The estimates remain conditional on this fixed task panel.', '',
             'PALDI has the highest final success among the six identification policies, but maximum-amplitude '
             'model-free control exceeds it with partial stimulation or Poisson evidence. Maximum-amplitude '
             'control succeeds on all valid partial-stimulation tasks, and on seven of eight uniform/Poisson '
             'tasks. These results do not preserve the manuscript advantage over this simple controller. '
             'Higher PALDI success under the new inputs is not, by itself, stronger evidence that learning '
             'is needed for this task. The earlier six-method table omitted these model-free references; '
             'their outcomes were simulated and are now reported explicitly below.', '',
             'The SNN already receives control at individual neurons. These input variants change which neurons '
             'receive that control and how sensory evidence reaches their synapses. The learner remains a '
             'two-state reduced model; removing its sigmoid does not remove that reduction.', '',
             'PALDI after 20 identification trials, control budget 12:', '',
             '| Input condition | Sigmoid median R² | Original median R² | Sigmoid success | Original success | Maximum-amplitude success |',
             '|---|---:|---:|---:|---:|---:|']
    for target in ('uniform', 'partial'):
        pred = [r2_summary[(r2_summary.arm == f'{family}_{target}') &
                          (r2_summary.policy == 'adaptive')].iloc[0]['median']
                for family in ('logistic', 'transfer')]
        for mode in ('current', 'poisson'):
            scores = [control_summary[(control_summary.arm == f'{family}_{target}_{mode}') &
                (control_summary.policy == 'adaptive') & (control_summary.checkpoint == 20) &
                (control_summary.budget == 12.)].iloc[0]['mean'] for family in ('logistic', 'transfer')]
            max_amplitude = [control_summary[(control_summary.arm == f'{family}_{target}_{mode}') &
                (control_summary.policy == 'front') & (control_summary.budget == 12.)].iloc[0]['mean']
                for family in ('logistic', 'transfer')]
            assert np.isclose(*max_amplitude)
            lines.append(f'| {target}, {mode} evidence | {pred[0]:.3f} | {pred[1]:.3f} | '
                         f'{scores[0]:.1%} | {scores[1]:.1%} | {max_amplitude[0]:.1%} |')
    lines += ['', 'Prediction is measured without sensory evidence and is shared across the two evaluation modes. '
              'Current-evidence success uses seven valid task seeds; Poisson-evidence success uses eight.', '',
              'PALDI paired differences (95% seed-bootstrap interval):', '',
              '| Contrast | Condition | Mean difference | 95% interval |', '|---|---|---:|---:|']
    for row in rd[rd.policy == 'adaptive'].itertuples():
        lines.append(f'| Original minus sigmoid R² | {row.condition} | {row.delta:+.3f} | {row.lo:+.3f} to {row.hi:+.3f} |')
    for row in cd[cd.policy == 'adaptive'].itertuples():
        lines.append(f'| Original minus sigmoid success | {row.condition} | {100*row.delta:+.1f} pp | '
                     f'{100*row.lo:+.1f} to {100*row.hi:+.1f} pp |')
    for row in input_diffs[input_diffs.policy == 'adaptive'].itertuples():
        lines.append(f'| Input minus uniform/current success, {row.family} | {row.condition} | '
                     f'{100*row.delta:+.1f} pp | {100*row.lo:+.1f} to {100*row.hi:+.1f} pp |')
    if history_rows and history_control:
        hp = next(row for row in history_rows if row['policy'] == 'adaptive')
        hc = next(row for row in history_control if row['policy'] == 'adaptive')
        lines += ['', '## Historical manuscript baseline', '',
            f'The fresh sigmoid baseline does not exactly reproduce the saved manuscript baseline. '
            f'PALDI median R² is {hp["new_median"]:.3f}, compared with {hp["old_median"]:.3f}; '
            f'its mean paired R² change is {hp["mean_paired_delta"]:+.4f}. '
            f'Control success is {hc["new_success"]:.1%}, compared with {hc["old_success"]:.1%}. '
            'All ablation contrasts above use the fresh baseline run in this campaign.', '',
            'The uniform held-out trajectories match the saved manuscript arrays exactly. Seed-0 readout '
            'calibration and initial draws also match, but the PALDI training trajectories later diverge. '
            'The cause of that divergence is unresolved. The two historical comparison CSV files retain '
            'the per-policy differences. Privileged-reference fitting is reconstructed, as described below.']
    lines += ['', '## Prediction after 20 trials', '', '| Arm | Policy | Median R² | IQR |', '|---|---|---:|---:|']
    for row in r2_summary.itertuples():
        lines.append(f'| {row.arm} | {LABELS[row.policy]} | {row.median:.3f} | {row.q25:.3f}–{row.q75:.3f} |')
    lines += ['', '## Change-of-mind success after 20 trials, budget 12', '', '| Arm | Policy | Mean success |', '|---|---|---:|']
    for row in control_summary[(control_summary.checkpoint == 20) & (control_summary.budget == 12)].itertuples():
        lines.append(f'| {row.arm} | {LABELS.get(row.policy,row.policy)} | {row.mean:.3f} |')
    reference_labels = {'prior':'Initial model', 'reduced_fit':'Privileged fitted model',
                        'none':'No control', 'spread':'Constant-amplitude', 'front':'Maximum-amplitude'}
    lines += ['', '## Reference controllers, budget 12', '',
              'These controls were run alongside all learned-model evaluations. No control applies zero current. '
              'Model-free rows contain one outcome per valid task seed, with no training-seed replication. '
              'Initial-model and fitted-model rows average over 20 calibrated readouts; the fitted model '
              'itself is fixed within each condition. All rows use the same valid task panel within a condition.', '',
              '| Arm | Reference | Mean success |', '|---|---|---:|']
    for arm in EVAL_ARMS:
        for policy in reference_policies:
            row = reference_summary[(reference_summary.arm == arm) &
                (reference_summary.policy == policy) &
                ((reference_summary.budget == 12.) | (reference_summary.policy == 'none'))].iloc[0]
            lines.append(f'| {arm} | {reference_labels[policy]} | {row["mean"]:.3f} |')
    lines += ['', '## Limits', '',
      '- The two-state reduction remains. Only the response function changes; five effective coefficients are learned.',
      '- Family-specific generic priors are retained from the pilot. The comparison includes their effect.',
      '- Partial stimulation targets 120 of 240 neurons per pool with unequal weights and matched total current. '
      'Squared-current dose and peak neuron current increase; see data/*/input_diagnostics.json.',
      '- Poisson evidence is matched to a nominal 6 pA mean at −55 mV, not at every membrane voltage. '
      'Its extra random draws change the network noise realization across evidence modes. '
      'input_differences_common_tasks.csv compares common valid task seeds.',
      '- Control intervals resample training seeds only and are conditional on the fixed task panel.',
      '- Privileged references use 48 independent training trajectories, 16 starts and 300 Adam iterations. '
      'The historical task-like fitting program was removed, so its schedule is reconstructed and recorded in code. '
      'No reference fit is selected on the control-test outcomes.',
      '- The manuscript R² routine replaces nonfinite rollout states with its existing numerical sentinel. '
      'Very poor or negative scores are retained. Training finiteness is checked separately.', '',
      'Reproduce: python -m experiments.tnsre.wong_wang_compare --root <campaign-results>', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'verification.json').write_text(json.dumps({'training_runs':480,'r2_estimates':len(r2),
        'control_sessions':len(task),'valid_control_sessions':len(valid),'source_sha256':sources,
        'duplicate_keys':0,'all_training_completed':True,'energy_constraints_passed':True,
        'numerical_controller_failures':len(numerical_failures)},indent=2))


if __name__ == '__main__':
    main()
