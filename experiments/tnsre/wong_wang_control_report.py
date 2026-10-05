"""Exact coverage checks and paired control comparisons for frozen study plans."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.tnsre.wong_wang_control_study import digest, load_models, planned_jobs, write_json


def verified_rows(models, root, plan):
    """Read all and only expected immutable-plan rows; reject missing/duplicate keys."""
    rows = []
    diagnostic = plan.get('worker_stage') == 'diagnostic'
    for arm in plan['arms']:
        jobs = planned_jobs(plan, load_models(models, arm), arm)
        if diagnostic:
            jobs = [j for j in jobs if j['unit']['theta'] is not None]
        expected = {j['key'] for j in jobs}
        paths = sorted((root/plan['name']/arm).glob('diagnostic_*.jsonl' if diagnostic else 'worker_*.jsonl'))
        actual = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
        keys = [r['key'] for r in actual]
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError(f'{arm}: expected={len(expected)}, rows={len(keys)}, '
                             f'missing={len(expected-set(keys))}, unexpected={len(set(keys)-expected)}')
        if any(r['plan_sha256'] != digest(plan) for r in actual):
            raise ValueError('Source plan hash mismatch')
        rows.extend(actual)
    return rows


def paired_matrix(frame, policy, training_seeds, task_seeds, endpoint):
    """A training-seed by task-seed array; broadcast unreplicated open-loop controls."""
    data = frame[frame.policy == policy]
    if set(data.seed) == {-1}:
        values = data.set_index('task_seed')[endpoint].reindex(task_seeds).to_numpy(float)
        out = np.broadcast_to(values, (len(training_seeds), len(task_seeds))).copy()
    else:
        out = data.pivot(index='seed', columns='task_seed', values=endpoint).reindex(
            index=training_seeds, columns=task_seeds).to_numpy(float)
    if not np.isfinite(out).all():
        raise ValueError(f'Incomplete paired endpoint for {policy}')
    return out


def simultaneous_comparisons(data, plan, *, draws=20000, seed=20261005):
    """Crossed paired bootstrap and simultaneous max-standardized-error intervals.

    Training seeds and task-noise seeds are independently resampled, retaining
    their pairing across methods. Open-loop controls are broadcast, not treated
    as independently re-simulated observations. All reported comparisons share
    one family-wise 95% interval calibration. Positive differences favor PALDI.
    """
    rng = np.random.default_rng(seed)
    comparisons, resampled = [], []
    train = plan['training_seeds']
    train_weights = rng.multinomial(len(train), np.full(len(train), 1/len(train)), size=draws)/len(train)
    all_task_seeds = plan['task_seeds']
    # One set of full-panel task draws is shared across arms. For arms with
    # invalid initial decisions, renormalize to the valid subset in each draw.
    task_counts = rng.multinomial(len(all_task_seeds),
                                 np.full(len(all_task_seeds), 1/len(all_task_seeds)), size=draws)
    for (arm, condition), group in data[data.valid == 1].groupby(['arm', 'condition']):
        if 'adaptive' not in set(group.policy):
            continue
        seeds = sorted(group.task_seed.unique())
        if len(train) < 2 or len(seeds) < 2:
            continue
        endpoint = 'success' if group.task.iloc[0] == 'overturn' else 'rmse'
        sign = 1. if endpoint == 'success' else -1.
        paldi = paired_matrix(group, 'adaptive', train, seeds, endpoint)
        weights = task_counts[:, [all_task_seeds.index(s) for s in seeds]].astype(float)
        empty = weights.sum(1) == 0
        weights[empty] = 1.
        weights /= weights.sum(1, keepdims=True)
        for policy in sorted(set(group.policy)-{'adaptive'}):
            baseline = paired_matrix(group, policy, train, seeds, endpoint)
            delta = sign*(paldi-baseline)
            effect = float(delta.mean())
            samples = np.einsum('bi,ij,bj->b', train_weights, delta, weights, optimize=True)
            comparisons.append(dict(arm=arm, condition=condition, baseline=policy,
                endpoint=endpoint, advantage=effect, paldi=float(paldi.mean()),
                baseline_mean=float(baseline.mean()), training_seeds=len(train), task_seeds=len(seeds),
                bootstrap_se=float(samples.std(ddof=1))))
            resampled.append(samples)
    if not comparisons:
        return pd.DataFrame()
    samples = np.column_stack(resampled)
    means = np.array([c['advantage'] for c in comparisons])
    standard_errors = np.array([c['bootstrap_se'] for c in comparisons])
    active = standard_errors > 1e-12
    critical = float(np.quantile(np.max(np.abs((samples[:, active]-means[active])/
        standard_errors[active]), axis=1), .95)) if active.any() else 0.
    for c in comparisons:
        half = critical*c['bootstrap_se']
        c.update(lower=c['advantage']-half, upper=c['advantage']+half,
                 simultaneous_critical=critical, family_size=len(comparisons), draws=draws,
                 degenerate_bootstrap=int(c['bootstrap_se'] <= 1e-12))
        # A degenerate bootstrap is not sufficient evidence of superiority.
        c['advantage_established'] = int(c['lower'] > 0 and not c['degenerate_bootstrap'])
    return pd.DataFrame(comparisons)


def report_diagnostic(rows, out):
    flat = []
    for r in rows:
        if not r['valid']:
            continue
        for i, name in enumerate(r['candidates']):
            flat.append({k: r[k] for k in ('arm', 'policy', 'seed', 'task_seed', 'condition', 'state_error')} |
                dict(candidate=name, actual_success=r['actual'][i]['success'],
                    actual_peak_gap=r['actual'][i]['peak_gap'], energy=r['actual'][i]['energy'],
                    predicted_soft=r['prediction']['filtered']['soft'][i],
                    predicted_hard=r['prediction']['filtered']['hard'][i],
                    true_onset_soft=r['prediction']['true_onset']['soft'][i],
                    true_onset_hard=r['prediction']['true_onset']['hard'][i]))
    frame = pd.DataFrame(flat)
    frame.to_csv(out/'diagnostic_all.csv', index=False)
    summary = frame.groupby(['arm', 'policy', 'condition', 'candidate']).agg(
        actual_success=('actual_success', 'mean'), predicted_soft=('predicted_soft', 'mean'),
        predicted_hard=('predicted_hard', 'mean'), true_onset_hard=('true_onset_hard', 'mean'),
        state_error=('state_error', 'mean'), n=('task_seed', 'size')).reset_index()
    summary.to_csv(out/'diagnostic_summary.csv', index=False)
    write_json(out/'verification.json', dict(rows=len(rows), valid=sum(r['valid'] for r in rows),
        complete_unique_coverage=True, candidates_replayed=['selected', 'front', 'spread']))
    return summary


def report(models, root, plan):
    rows = verified_rows(models, root, plan)
    out = root/plan['name']/'comparison'
    out.mkdir(parents=True, exist_ok=True)
    if plan.get('worker_stage') == 'diagnostic':
        return report_diagnostic(rows, out)
    data = pd.DataFrame(rows)
    numeric = ['energy', 'peak_action_norm', 'peak_neuron_current_pa', 'total_squared_current_pa2_s']
    assert np.isfinite(data[numeric]).all().all()
    assert (data.energy <= data.budget+1e-6).all()
    assert (data.peak_action_norm <= data.bound+1e-6).all()
    assert data.groupby(['arm', 'task_seed']).valid.nunique().max() == 1
    tracking = data[(data.valid == 1) & (data.task != 'overturn')]
    assert np.isfinite(tracking[['rmse', 'occupancy']]).all().all()
    data.to_csv(out/'control_all.csv', index=False)
    valid = data[data.valid == 1]
    by_seed = valid.groupby(['arm', 'condition', 'policy', 'seed'], as_index=False).agg(
        success=('success', 'mean'), rmse=('rmse', 'mean'), occupancy=('occupancy', 'mean'),
        energy=('energy', 'mean'), neuron_dose=('total_squared_current_pa2_s', 'mean'),
        failures=('numerical_failure', 'sum'), n_tasks=('task_seed', 'size'))
    by_seed.to_csv(out/'control_by_seed.csv', index=False)
    summary = by_seed.groupby(['arm', 'condition', 'policy'], as_index=False).agg(
        success=('success', 'mean'), rmse=('rmse', 'mean'), occupancy=('occupancy', 'mean'),
        energy=('energy', 'mean'), neuron_dose=('neuron_dose', 'mean'),
        failures=('failures', 'sum'), n_training_seeds=('seed', 'size'), n_tasks=('n_tasks', 'first'))
    summary.to_csv(out/'control_summary.csv', index=False)
    comparisons = simultaneous_comparisons(data, plan)
    comparisons.to_csv(out/'paired_comparisons.csv', index=False)
    verified = dict(rows=len(data), valid=int(data.valid.sum()), complete_unique_coverage=True,
        numerical_failures=int(data.numerical_failure.sum()), bounds_and_budgets_passed=True,
        plan_sha256=digest(plan), all_baseline_advantage=False)
    if len(comparisons):
        verified['all_baseline_advantage'] = bool(comparisons.advantage_established.all())
    write_json(out/'verification.json', verified)
    lines = [f'# {plan["name"]}', '', f'Complete: {len(data):,} evaluations; '
        f'{verified["numerical_failures"]} numerical controller failures.', '',
        'These results are conditional on the frozen identification models and fixed network/targeting masks. '
        'Open-loop controls have one result per task seed, without training-seed replication. '
        'Tracking RMSE is primary; success requires 90% occupancy in the preset target band.', '']
    for (arm, condition), group in summary.groupby(['arm', 'condition']):
        lines.extend([f'## {arm}: {condition}', '',
            '| Method | Success | Tracking RMSE | Band occupancy | Mean energy |',
            '|---|---:|---:|---:|---:|'])
        for r in group.itertuples():
            rmse = '—' if pd.isna(r.rmse) else f'{r.rmse:.3f}'
            occupancy = '—' if pd.isna(r.occupancy) else f'{100*r.occupancy:.1f}%'
            lines.append(f'| {r.policy} | {100*r.success:.1f}% | {rmse} | {occupancy} | {r.energy:.3f} |')
        lines.append('')
    lines.append('Paired comparisons use simultaneous 95% crossed-bootstrap intervals across the reported family. '
                 'Smoke runs with a single training seed do not establish an advantage.')
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


def select_baselines(models, root, plan):
    """Freeze pulse/relay/PI settings using a complete calibration panel only.

    Minimize mean trial RMSE for tracking; maximize success for overturn. Ties
    use mean energy over all valid trials, then the candidate label. These
    choices never inspect PALDI or confirmation outcomes. Identical physical
    inputs/readouts permit sharing baseline settings across learner families.
    """
    if plan.get('purpose') != 'baseline_tuning' or any(p not in ('none', 'front', 'spread')
                                                    for p in plan['policies']):
        raise ValueError('Baseline selection requires a dedicated model-free calibration plan')
    rows = verified_rows(models, root, plan)
    data = pd.DataFrame(rows)
    specifications = {s['label']: s for s in plan['baselines']}
    choices, baselines = [], []
    for (arm, condition), group in data[data.valid == 1].groupby(['arm', 'condition']):
        for controller in ('pulse', 'relay', 'pi'):
            candidates = group[group.controller == controller]
            if candidates.empty:
                raise ValueError(f'No calibrated {controller} for {arm}/{condition}')
            scores = candidates.groupby('policy').agg(
                success=('success', 'mean'), rmse=('rmse', 'mean'), energy=('energy', 'mean'))
            tracking = group.task.iloc[0] != 'overturn'
            scores['loss'] = scores.rmse if tracking else -scores.success
            scores = scores.reset_index().sort_values(['loss', 'energy', 'policy'])
            best = scores.iloc[0]
            spec = specifications[best.policy]
            suffix = arm.split('_', 1)[1]
            matching = [f'{family}_{suffix}' for family in ('logistic', 'transfer')]
            # Check readout identity rather than assuming it from the arm names.
            family_units = [{u['seed']: u for u in load_models(models, a) if u['policy']=='prior'}
                            for a in matching]
            for seed in family_units[0]:
                for key in ('C', 'b'):
                    np.testing.assert_allclose(family_units[0][seed][key], family_units[1][seed][key],
                                               rtol=0, atol=0)
            baselines.append(dict(label=f'{controller}_tuned', controller=controller,
                parameters=spec['parameters'], arms=matching, conditions=[condition]))
            choices.append(dict(arm=arm, condition=condition, controller=controller,
                selected=best.policy, parameters=spec['parameters'], loss=float(best.loss),
                energy=float(best.energy), candidates=len(scores)))
    out = root/plan['name']/'comparison'
    write_json(out/'selected_baselines.json', dict(source_plan_sha256=digest(plan),
        source_task_seeds=plan['task_seeds'], source_training_seeds=plan['training_seeds'],
        selection='Mean calibration endpoint; ties lower mean energy then label',
        baselines=baselines, choices=choices))
    return baselines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--select-baselines', action='store_true')
    a = p.parse_args()
    result = report(a.models, a.root, json.loads(a.plan.read_text()))
    print(result.to_string(index=False))
    if a.select_baselines:
        select_baselines(a.models, a.root, json.loads(a.plan.read_text()))


if __name__ == '__main__':
    main()
