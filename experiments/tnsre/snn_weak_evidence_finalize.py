"""Assemble, verify, and compare the weak evidence-trial control experiment (no simulation).

Input: the collected GCP run of ``experiments.tnsre.snn_weak_evidence`` and the frozen selected
result (strong control throughout). The output folder follows the selected result's layout,
so the manuscript figure generator runs on it unchanged:

* ``tracks/``, ``eval/`` (prediction, task rows, summaries), ``data/``, ``reference/`` (frozen
  held-out data and fit), ``control_phase/``.
* ``comparison/``: paired weak-versus-strong and PALDI-versus-method contrasts, evidence-trial
  diagnostics against the evidence-only replay, bound compliance, and trial durations.
* ``provenance/``: VM ledgers, run checks, replay validation, cloud logs, verification records.

Reference task rows (prior, fitted model, model-free inputs) are the frozen rows; ``verify``
checks identical readouts and priors, and that the VM's example replay reproduces them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre import snn_weak_evidence as w
from experiments.tnsre.wong_wang_full import write_json

FROZEN = Path('results/archive/experiments/20261007_snn_native_uniform')
FROZEN_RUNS = FROZEN / 'tracks' / 'ww_observation_transfer_uniform'
LEARNED = ('adaptive', 'active_myopic', 'flex_rollback', 'rhc', 'prbs', 'random')
REFERENCES = ('prior', 'reduced_fit', 'none', 'spread', 'front')
TASK_SEEDS = list(range(91000, 91007))  # 91007 has no initial decision and is excluded for all
SCHEDULES = {'strong': FROZEN_RUNS, 'weak': None}  # weak runs: <out>/tracks/<slug>
WEAK = w.SCENARIOS['weak2_e10']
# Settings that must be identical between matched weak and strong runs.
MATCHED_SETTINGS = ('policy_id', 'seed', 'action_max', 'action_cost_weight', 'planning_horizon', 'replan_interval',
                    'update_interval', 'schedule_id', 'coarse_dt_factor', 'coarse_action_mapping', 'adaptive_cadence',
                    'adaptive_replan_min_interval', 'adaptive_update_min_interval', 'adaptive_update_eig_threshold',
                    'adaptive_replan_state_error_threshold', 'estimator_dynamics_type', 'initial_parameter',
                    'initial_parameter_variance', 'parameter_prior_covariance', 'q_theta', 'dt', 'state_noise',
                    'total_steps', 'observation_loading_matrix', 'observation_loading_bias', 'flex_regularization',
                    'flex_parameter_step_clip', 'flex_parameter_min', 'flex_parameter_max', 'planning_rollout',
                    'learning_sensitivity', 'estimator_true_params_full', 'min_embedding_dim')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weak_runs(out: Path) -> Path:
    return w.runs_root(out, WEAK)


def learned_policies(scenario: w.Scenario) -> tuple[str, ...]:
    """Evaluation names of a scenario's identification policies (PALDI variants count as PALDI)."""
    return tuple(ev.evaluated_policy(p) for p in scenario.policies)


def frozen_twin(policy_dir: str) -> str:
    """Frozen run with the same learner settings; the passive learner shares Random's update schedule."""
    return policy_dir if policy_dir in w.POLICIES else 'random'


def configure(out: Path):
    w.configure_eval(out / 'data')
    ev.PROTOCOL['reduced_fit'] = str(out / 'reference' / 'native_fit.json')


# ------------------------------------------------------------------ assemble
def assemble(run: Path, out: Path, source: Path | None = None):
    """Copy one scenario of the collected run (``gcp_runs/<id>``) and the frozen inputs into the result layout.

    ``source`` is the scenario root inside the run (default: the 2026-10-07 layout). VM ledgers
    and logs are copied when they are inside it; a multi-scenario campaign keeps them one level up.
    """
    source = run / 'results' / 'weak_evidence' if source is None else source
    assert not out.exists() or not any(out.iterdir()), f'{out} is not empty; finalize writes a new folder'
    shutil.copytree(source / 'tracks', out / 'tracks')
    (out / 'data').mkdir()
    for name in ('test_trajectories.npz', 'reference_train.npz', 'input_diagnostics.json'):
        shutil.copy2(FROZEN / 'data' / name, out / 'data' / name)
    (out / 'reference').mkdir()
    shutil.copy2(FROZEN / 'reference' / 'native_fit.json', out / 'reference' / 'native_fit.json')
    groups = sorted(source.glob('eval/group_*'))
    shutil.copytree(source / 'eval', out / 'eval' / 'groups')
    for name in ('r2_per_session.csv', 'identification_sessions.csv'):
        frame = pd.concat([pd.read_csv(g / name) for g in groups], ignore_index=True)
        frame.sort_values(['policy', 'seed'] + (['sessions'] if 'sessions' in frame else ['session']),
                          kind='stable').to_csv(out / 'eval' / name, index=False)
    task = out / 'eval' / 'task_warm'; task.mkdir()
    for path in sorted(source.glob('eval/group_*/task_warm/worker_*.csv')):
        shutil.copy2(path, task / path.name)
    for path in sorted(source.glob('eval/group_*/failures/*.json')):
        (out / 'eval' / 'failures').mkdir(exist_ok=True)
        shutil.copy2(path, out / 'eval' / 'failures' / path.name)
    frozen = pd.concat([pd.read_csv(p) for p in sorted((FROZEN / 'eval' / 'task_warm').glob('worker_*.csv'))],
                       ignore_index=True)
    frozen[frozen.policy.isin(REFERENCES)].to_csv(task / 'worker_frozen_references.csv', index=False)
    if (source / 'examples_replay').exists():
        shutil.copytree(source / 'examples_replay' / 'examples', out / 'eval' / 'examples')
        shutil.copytree(source / 'examples_replay' / 'control_phase', out / 'control_phase')
    shutil.copytree(source / 'evidence_only', out / 'evidence_only')
    vm = out / 'provenance' / 'vm'
    shutil.copytree(source / 'run_checks', vm / 'run_checks')
    if (source / 'logs').exists():
        shutil.copytree(source / 'logs', vm / 'logs')
    for path in sorted(source.glob('*_group_*.json')):
        shutil.copy2(path, vm / path.name)
    shutil.copy2(run / 'commit.txt', out / 'provenance' / 'commit.txt')
    shutil.copytree(run / 'logs', out / 'provenance' / 'cloud_logs')
    write_json(out / 'provenance' / 'inputs.json', dict(
        frozen_result=str(FROZEN), source_run=str(run),
        files={str(p.relative_to(out)): sha256(p) for p in [*sorted((out / 'data').iterdir()),
                                                              out / 'reference' / 'native_fit.json']},
        frozen_files={str(p.relative_to(FROZEN)): sha256(p) for p in [*sorted((FROZEN / 'data').iterdir()),
                                                                       FROZEN / 'reference' / 'native_fit.json']}))


def protocol(out: Path):
    """``eval/protocol.json``: the evaluation protocol shared by every VM group (paths to this folder)."""
    groups = [json.loads(p.read_text()) for p in sorted((out / 'eval' / 'groups').glob('group_*/protocol.json'))]
    local = dict(test_trajectories=str(out / 'data' / 'test_trajectories.npz'),
                 reduced_fit=str(out / 'reference' / 'native_fit.json'))
    merged = [{**g, 'protocol': {**g['protocol'], **local}} for g in groups]
    assert len(merged) == 4 and all(m == merged[0] for m in merged), 'VM groups used different protocols'
    write_json(out / 'eval' / 'protocol.json', {**merged[0], 'source': 'eval/groups/group_*/protocol.json'})


# ------------------------------------------------------------------ verify
def readout_hash(run: dict) -> str:
    """The frozen training verification's readout key: sha256 of (C.ravel(), b) bytes."""
    return hashlib.sha256(np.concatenate([run['C'].ravel(), run['b']]).tobytes()).hexdigest()


# Session-rule fields a scenario may change; every other field equals the frozen rule.
SCENARIO_RULE_FIELDS = {'action_scale', 'action_scale_last_session', 'action_scale_later', 'evidence_last_session',
                        'fixed_first_session'}
# Rule fields the frozen runs predate, with the value that reproduces their protocol.
RULE_DEFAULTS = {'action_scale_later': 1.0}


def verify(out: Path, scenario: w.Scenario = WEAK):
    """Matched configuration, bound compliance, exact task grid, and reusable frozen references."""
    configure(out)
    runs_root, learned = w.runs_root(out, scenario), learned_policies(scenario)
    new_runs, strong = ev.load_session_runs(runs_root), ev.load_session_runs(FROZEN_RUNS)
    assert {(r['policy'], r['seed']) for r in new_runs} == {(p, s) for p in learned for s in w.SEEDS}
    assert {(r['policy'], r['seed']) for r in strong} == {(p, s) for p in LEARNED for s in w.SEEDS}
    frozen_readouts = json.loads((FROZEN / 'provenance' / 'training_verification.json').read_text())['readouts_by_seed']
    by_key = {(r['policy'], r['seed']): r for r in strong}
    for r in new_runs:
        assert set(r['est']) == set(range(21)), (r['policy'], r['seed'])
        assert readout_hash(r) == frozen_readouts[str(r['seed'])]
        twin = r['policy'] if r['policy'] in LEARNED else 'random'  # see frozen_twin
        np.testing.assert_array_equal(r['est'][0], by_key[(twin, r['seed'])]['est'][0])
    settings, rule_changes = [], set()
    for path in sorted(runs_root.glob('*/seed_*/repeat_01/run_metadata.json')):
        meta = json.loads(path.read_text())
        rel = path.relative_to(runs_root)
        twin = frozen_twin(rel.parts[0])
        old = json.loads((FROZEN_RUNS / twin / Path(*rel.parts[1:])).read_text())
        matched = [k for k in MATCHED_SETTINGS if k in old and not (k == 'policy_id' and twin != rel.parts[0])]
        differ = [k for k in matched if meta.get(k) != old[k]]
        assert not differ, (path, differ)
        rule, old_rule = meta['session_rule'], old['session_rule']
        changed = {k for k in rule if rule[k] != old_rule.get(k, RULE_DEFAULTS.get(k))}
        assert set(old_rule) <= set(rule) and changed <= SCENARIO_RULE_FIELDS, (rule, old_rule)
        assert (rule['action_scale'], rule['action_scale_last_session']) == (scenario.scale, scenario.weak_trials)
        assert rule.get('action_scale_later', 1.0) == scenario.later_scale
        assert rule['evidence_last_session'] == rule['fixed_first_session'] == scenario.evidence_trials
        assert meta['env_preset_id'] == scenario.preset and old['env_preset_id'] == w.TASK_PRESET
        settings.append(len(matched))
        rule_changes |= changed
    checks = [json.loads(p.read_text()) for p in sorted((out / 'provenance' / 'vm' / 'run_checks').glob('*.json'))]
    assert len(checks) == 20 * len(learned) and all(c['compliant'] and c['evidence_max_error'] <= 1e-6 for c in checks)
    replay = [json.loads(p.read_text()) for p in sorted((out / 'evidence_only').glob('validation_group_*.json'))]
    # Task grid: new learned rows plus frozen reference rows. Every learned policy has the
    # frozen PALDI grid (same cells and task seeds); the references are the frozen rows.
    rows = pd.concat([pd.read_csv(p) for p in sorted((out / 'eval' / 'task_warm').glob('worker_*.csv'))],
                     ignore_index=True)
    frozen = pd.concat([pd.read_csv(p) for p in sorted((FROZEN / 'eval' / 'task_warm').glob('worker_*.csv'))],
                       ignore_index=True)
    keys = lambda f: set(f[w.KEY].itertuples(index=False, name=None))
    grid = keys(frozen[frozen.policy == 'adaptive'])
    expected = keys(frozen[frozen.policy.isin(REFERENCES)]) | {(c, p, *rest) for p in learned for c, _, *rest in grid}
    assert not rows.duplicated(w.KEY).any() and keys(rows) == expected
    new = rows[rows.policy.isin(learned)]
    assert len(new) == len(grid) * len(learned) and len(grid) == 1280
    assert np.isfinite(new.energy).all() and (new.energy <= new.budget + 1e-6).all()
    assert set(rows.loc[rows.valid == 0, 'task_seed']) == {91007} and (rows.valid == (rows.task_seed != 91007)).all()
    for _, row in new[new.numerical_failure == 1].iterrows():
        assert (out / 'eval' / 'failures' / Path(row.failure_record).name).exists()
    # The VM's example replay (scenarios with PALDI) must reproduce the frozen fitted-model and model-free rows.
    examples = (pd.read_csv(out / 'eval' / 'examples' / 'example_sessions.csv')
                if (out / 'eval' / 'examples').exists() else pd.DataFrame(columns=['controller']))
    reproduced = []
    for _, row in examples[examples.controller != 'learned'].iterrows():
        policy = {'fit': 'reduced_fit'}.get(row.controller, row.controller)
        budget = 10. if policy == 'none' else row.budget
        match = frozen[(frozen.policy == policy) & (frozen.seed == (0 if policy == 'reduced_fit' else -1))
                       & (frozen.budget == budget) & (frozen.task_seed == row.task_seed)]
        assert len(match) == 1
        for key in ['valid', 'target', 'onset', 'success', 'outcome_bin', 'energy']:
            np.testing.assert_allclose(row[key], match.iloc[0][key], rtol=1e-5, atol=1e-6)
        reproduced.append(policy)
    constraints = {}
    for path in sorted(runs_root.glob('*/seed_*/repeat_01/run_metadata.json')):
        constraints.setdefault(path.parts[-4], set()).add(json.loads(path.read_text()).get('action_constraint', 'box'))
    assert all(len(v) == 1 for v in constraints.values()), constraints
    record = dict(passed=True, scenario=dict(preset=scenario.preset, policies=list(scenario.policies),
                                             scale=scenario.scale, weak_trials=scenario.weak_trials,
                                             evidence_trials=scenario.evidence_trials,
                                             constraint=scenario.constraint, later_scale=scenario.later_scale),
                  action_constraint_by_policy={k: v.pop() for k, v in constraints.items()},
                  runs=len(new_runs), readouts_match_frozen=True, priors_match_frozen=True,
                  matched_settings_per_run=sorted(set(settings)), session_rule_fields_changed=sorted(rule_changes),
                  run_checks=len(checks),
                  replay_validation=[dict(group=i, passed=v['passed']) for i, v in enumerate(replay)],
                  task_rows=len(rows), learned_rows=len(new), frozen_reference_rows=len(rows) - len(new),
                  numerical_failures=int(new.numerical_failure.sum()),
                  example_reproduces_frozen_references=reproduced,
                  frozen_task_sha256={p.name: sha256(p) for p in sorted((FROZEN / 'eval' / 'task_warm').glob('worker_*.csv'))})
    write_json(out / 'provenance' / 'verification.json', record)
    print(json.dumps({k: v for k, v in record.items() if k != 'frozen_task_sha256'}, indent=1))


# ------------------------------------------------------------------ score
def score(out: Path, scenario: w.Scenario = WEAK):
    """Summaries and the main table in the selected result's format."""
    configure(out)
    learned = learned_policies(scenario)
    ev.stage_score(out / 'eval', 'warm')
    summary = pd.read_csv(out / 'eval' / 'task_summary_warm.csv')
    final = summary[(summary.checkpoint == 20) | summary.policy.isin(REFERENCES)]
    final.to_csv(out / 'all_baselines.csv', index=False)
    r2 = pd.read_csv(out / 'eval' / 'r2_per_session.csv')
    records = []
    for policy in (*learned, *REFERENCES):
        record = dict(policy=policy)
        if policy in learned or policy == 'prior':
            subset = r2[(r2.policy == (learned[0] if policy == 'prior' else policy))
                        & (r2.sessions == (0 if policy == 'prior' else 20))]
            record.update(zip(['r2_q25', 'r2_median', 'r2_q75'], np.percentile(subset.r2, [25, 50, 75])))
        elif policy == 'reduced_fit':
            record['r2_median'] = json.loads((out / 'reference' / 'native_fit.json').read_text())['r2_test_piecewise_current']
        for budget in (5, 10, 12, 15):
            cell = final[(final.policy == policy) & (final.budget == (10 if policy == 'none' else budget))]
            assert len(cell) == 1
            record.update({f'success_{40 * budget}_pa2s': float(cell.iloc[0].success),
                           f'sem_{40 * budget}_pa2s': float(cell.iloc[0].sem_over_seeds)})
        records.append(record)
    pd.DataFrame(records).to_csv(out / 'main_table.csv', index=False)


# ------------------------------------------------------------------ paired contrasts
def success_matrix(rows: pd.DataFrame, policy: str, checkpoint: int, budget: float) -> np.ndarray:
    """Training seed x valid task seed success, shape (20, 7); model-free rows broadcast over seeds."""
    data = rows[(rows.policy == policy) & (rows.budget == (10. if policy == 'none' else budget))
                & (rows.task_seed.isin(TASK_SEEDS))]
    if policy not in REFERENCES:
        data = data[data.checkpoint == checkpoint]
    if set(data.seed) == {-1}:
        values = data.set_index('task_seed').success.reindex(TASK_SEEDS).to_numpy(float)
        return np.broadcast_to(values, (len(w.SEEDS), len(TASK_SEEDS))).copy()
    out = data.pivot(index='seed', columns='task_seed', values='success').reindex(
        index=w.SEEDS, columns=TASK_SEEDS).to_numpy(float)
    assert np.isfinite(out).all(), policy
    return out


def r2_vector(r2: pd.DataFrame, policy: str, sessions: int) -> np.ndarray:
    data = r2[(r2.policy == policy) & (r2.sessions == sessions)].set_index('seed').r2.reindex(w.SEEDS)
    assert data.notna().all()
    return data.to_numpy(float)


def simultaneous(contrasts: list[dict], *, draws: int = 20000, seed: int = 20261007) -> pd.DataFrame:
    """Paired training-seed bootstrap with one simultaneous 95% interval per endpoint class.

    As in the selected result's audit (``results/archive/experiments/20261007_wong_wang_comparisons/
    updated_observation_20261006/analyze.py``): the 20 identification seeds are resampled together
    for every contrast; control intervals are conditional on the fixed task panel. R2 contrasts
    pair vectors (20,) and use the difference of medians; success contrasts pair matrices (20, 7)
    and use the mean paired difference. A max-standardized-deviation calibration over the
    contrasts of one endpoint gives their family-wise interval.
    """
    rng = np.random.default_rng(seed)
    train = rng.integers(0, len(w.SEEDS), size=(draws, len(w.SEEDS)))
    samples, rows = [], []
    for c in contrasts:
        a, b = c.pop('a'), c.pop('b')
        if a.ndim == 1:
            estimate = float(np.median(a) - np.median(b))
            sample = np.median(a[train], axis=1) - np.median(b[train], axis=1)
            statistic = 'difference of medians'
        else:
            per_seed = (a - b).mean(axis=1)
            estimate, sample = float(per_seed.mean()), per_seed[train].mean(axis=1)
            statistic = 'mean paired difference'
        rows.append(dict(**c, value_a=float(np.median(a) if a.ndim == 1 else a.mean()),
                         value_b=float(np.median(b) if b.ndim == 1 else b.mean()), statistic=statistic,
                         difference=estimate, bootstrap_se=float(sample.std(ddof=1))))
        samples.append(sample)
    frame, samples = pd.DataFrame(rows), np.column_stack(samples)
    for endpoint, index in frame.groupby('endpoint').groups.items():
        index = np.asarray(list(index))
        means, se = frame.difference.to_numpy()[index], frame.bootstrap_se.to_numpy()[index]
        active = se > 1e-12
        critical = float(np.quantile(np.max(np.abs((samples[:, index[active]] - means[active]) / se[active]), axis=1), .95))
        half = critical * se
        frame.loc[index, 'lower'], frame.loc[index, 'upper'] = means - half, means + half
        frame.loc[index, 'simultaneous_critical'], frame.loc[index, 'family_size'] = critical, len(index)
    frame['draws'], frame['bootstrap_seed'] = draws, seed
    frame['interval_excludes_zero'] = ((frame.bootstrap_se > 1e-12) & ((frame.lower > 0) | (frame.upper < 0))).astype(int)
    return frame


def task_rows(path: Path) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in sorted(path.glob('worker_*.csv'))], ignore_index=True)


def compare(out: Path):
    """Weak minus strong per method (paired by seed), and PALDI minus each method within each schedule."""
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    r2 = {'strong': pd.read_csv(FROZEN / 'eval' / 'r2_per_session.csv'), 'weak': pd.read_csv(out / 'eval' / 'r2_per_session.csv')}
    rows = {'strong': task_rows(FROZEN / 'eval' / 'task_warm'), 'weak': task_rows(out / 'eval' / 'task_warm')}
    schedule = []
    for policy in LEARNED:
        for sessions in (10, 20):
            schedule.append(dict(contrast='weak - strong', policy=policy, endpoint='prediction_R2', trials=sessions,
                                 budget_pa2_s=np.nan, a=r2_vector(r2['weak'], policy, sessions),
                                 b=r2_vector(r2['strong'], policy, sessions)))
        for budget in (10., 12.):
            schedule.append(dict(contrast='weak - strong', policy=policy, endpoint='overturn_success', trials=20,
                                 budget_pa2_s=40 * budget, a=success_matrix(rows['weak'], policy, 20, budget),
                                 b=success_matrix(rows['strong'], policy, 20, budget)))
    simultaneous(schedule).to_csv(comparison / 'weak_minus_strong.csv', index=False)
    within = []
    for name in ('weak', 'strong'):
        for policy in LEARNED[1:]:
            within.append(dict(contrast=f'PALDI - method ({name})', policy=policy, endpoint='prediction_R2', trials=20,
                               budget_pa2_s=np.nan, a=r2_vector(r2[name], 'adaptive', 20), b=r2_vector(r2[name], policy, 20)))
        for budget in (10., 12.):
            for policy in (*LEARNED[1:], *REFERENCES):
                within.append(dict(contrast=f'PALDI - method ({name})', policy=policy, endpoint='overturn_success',
                                   trials=20, budget_pa2_s=40 * budget, a=success_matrix(rows[name], 'adaptive', 20, budget),
                                   b=success_matrix(rows[name], policy, 20, budget)))
    simultaneous(within).to_csv(comparison / 'paldi_minus_method.csv', index=False)
    # Descriptive table: prediction quartiles and mean success per schedule.
    table = []
    for policy in LEARNED:
        record = dict(policy=policy)
        for name in ('strong', 'weak'):
            for sessions in (10, 20):
                q = np.percentile(r2_vector(r2[name], policy, sessions), [25, 50, 75])
                record.update({f'{name}_r2_t{sessions}_q25': q[0], f'{name}_r2_t{sessions}_median': q[1],
                               f'{name}_r2_t{sessions}_q75': q[2]})
            for budget in (5., 10., 12., 15.):
                record[f'{name}_success_{40 * budget:g}_pa2s'] = float(success_matrix(rows[name], policy, 20, budget).mean())
            record[f'{name}_success_480_trial10'] = float(success_matrix(rows[name], policy, 10, 12.).mean())
        table.append(record)
    pd.DataFrame(table).to_csv(comparison / 'summary.csv', index=False)


def audit(out: Path, scenario: w.Scenario, seed: int = 20261008):
    """PALDI minus every other method within one schedule: the paired audit of a single result.

    The family of ``compare``'s ``paldi_minus_method`` for one schedule: R2 after trial 20 against
    the five learned baselines, and success at 400 and 480 pA^2 s after trial 20 against them and
    every reference (5 and 20 contrasts, one simultaneous interval per endpoint).
    """
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    r2, rows = pd.read_csv(out / 'eval' / 'r2_per_session.csv'), task_rows(out / 'eval' / 'task_warm')
    paldi, *others = learned_policies(scenario)
    assert paldi == 'adaptive'
    contrasts = [dict(contrast='PALDI - method', policy=p, endpoint='prediction_R2', trials=20, budget_pa2_s=np.nan,
                      a=r2_vector(r2, paldi, 20), b=r2_vector(r2, p, 20)) for p in others]
    for budget in (10., 12.):
        contrasts += [dict(contrast='PALDI - method', policy=p, endpoint='overturn_success', trials=20,
                           budget_pa2_s=40 * budget, a=success_matrix(rows, paldi, 20, budget),
                           b=success_matrix(rows, p, 20, budget)) for p in (*others, *REFERENCES)]
    frame = simultaneous(contrasts, seed=seed)
    frame.to_csv(comparison / 'paldi_minus_method.csv', index=False)
    return frame


# ------------------------------------------------------------------ one result from a campaign scenario
def assemble_result(run: Path, out: Path, name: str):
    """``assemble`` of one scenario of a collected campaign, plus the campaign's VM ledgers and logs."""
    campaign = run / 'results' / 'evidence_scenarios'
    assemble(run, out, source=campaign / name)
    vm = out / 'provenance' / 'vm'
    shutil.copytree(campaign / 'logs', vm / 'logs')
    for path in sorted(campaign.glob('*_group_*.json')):
        shutil.copy2(path, vm / path.name)


def evidence(out: Path, scenario: w.Scenario):
    """Evidence-led trials against this result's evidence-only replay, trial time, and bound compliance."""
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    replay = pd.concat([pd.read_csv(p) for p in sorted((out / 'evidence_only').glob('group_*.csv'))], ignore_index=True)
    runs_root = w.runs_root(out, scenario)
    trials = evidence_trials(runs_root, 'result', replay, scenario.evidence_trials).drop(columns='schedule')
    assert trials.groupby('policy').size().eq(20 * scenario.evidence_trials).all()
    trials.to_csv(comparison / 'evidence_trials.csv', index=False)
    summary = trials.groupby('policy', sort=False).agg(
        trials=('trial', 'size'), decisions=('chosen', lambda c: int((c >= 0).sum())),
        first_choice_against_evidence=('against_evidence', 'sum'),
        choice_flipped_vs_evidence_only=('choice_flipped_vs_evidence_only', 'sum'),
        recorded_postdecision_reversals=('recorded_postdecision_reversal', 'sum'),
        median_decision_ms=('decision_ms', 'median'), evidence_only_median_decision_ms=('evidence_only_decision_ms', 'median'),
        control_rms_pa=('control_rms_pa', 'mean')).reset_index()
    time = []
    for path in sorted(runs_root.glob('*/seed_*/repeat_01/run_metadata.json')):
        meta = json.loads(path.read_text())
        bins = np.array([t['bins'] for t in meta['session_log']])
        time.append(dict(policy=ev.evaluated_policy(meta['policy_id']), identification_s=0.005 * bins.sum(),
                         evidence_trials_s=0.005 * bins[:scenario.evidence_trials].sum()))
    time = pd.DataFrame(time).groupby('policy')[['identification_s', 'evidence_trials_s']].mean().reset_index()
    summary.merge(time, on='policy', validate='one_to_one').to_csv(comparison / 'evidence_summary.csv', index=False)
    checks = [json.loads(p.read_text()) for p in sorted((out / 'provenance' / 'vm' / 'run_checks').glob('*.json'))]
    pd.DataFrame([dict(policy=ev.evaluated_policy(c['policy']), seed=c['seed'], geometry=c['geometry'],
                       action_constraint=c.get('action_constraint'), trials=v['trials'], bound_pa=20 * v['bound'],
                       max_size_pa=20 * v['max_size'], max_ratio=v['max_ratio'], mean_ratio=v['mean_ratio'],
                       at_bound_fraction=v['at_bound_fraction'])
                  for c in checks for v in c['phases'].values()]).to_csv(comparison / 'bound_compliance.csv', index=False)


def versus(out: Path, scenario: w.Scenario, other: Path, other_scenario: w.Scenario,
           identical: list[str] | None = None, seed: int = 20261008):
    """This result against another result of the same evaluation: another identification
    constraint (``Scenario.constraint``), bound schedule, or code version.

    Policies in ``identical`` must give identical models, R2 and task rows. By default these are
    the policies whose resolved catalog entry and identification protocol agree under both
    scenarios; a code change between the runs is not visible there and needs ``identical``. The
    others get paired training-seed contrasts (this minus other): R2 after trial 20, and success
    at 400 and 480 pA^2 s after trial 20. Outputs are named after the other result's folder.
    """
    if identical is None:
        specs = {c: w.configure_catalogs(c).policy_specs for c in (scenario.constraint, other_scenario.constraint)}
        protocol = lambda sc: (sc.preset, sc.scale, sc.weak_trials, sc.evidence_trials, sc.later_scale)
        identical = [p for p in scenario.policies if specs[scenario.constraint][p] == specs[other_scenario.constraint][p]
                     and protocol(scenario) == protocol(other_scenario)]
    assert set(identical) <= set(scenario.policies), identical
    same = [p for p in scenario.policies if p in identical]
    changed = [p for p in scenario.policies if p not in same]
    w.configure_catalogs(scenario.constraint)
    name = {p: ev.evaluated_policy(p) for p in scenario.policies}
    runs = {k: {(r['policy'], r['seed']): r for r in ev.load_session_runs(w.runs_root(d, sc))}
            for k, d, sc in (('this', out, scenario), ('other', other, other_scenario))}
    r2 = {'this': pd.read_csv(out / 'eval' / 'r2_per_session.csv'), 'other': pd.read_csv(other / 'eval' / 'r2_per_session.csv')}
    rows = {'this': task_rows(out / 'eval' / 'task_warm'), 'other': task_rows(other / 'eval' / 'task_warm')}
    checked = {}
    for p in (name[p] for p in same):
        for s in w.SEEDS:
            a, b = runs['this'][(p, s)]['est'], runs['other'][(p, s)]['est']
            assert all(np.array_equal(a[k], b[k]) for k in range(21)), (p, s)
        a, b = (r2[k][r2[k].policy == p].sort_values(['seed', 'sessions']).reset_index(drop=True) for k in r2)
        pd.testing.assert_frame_equal(a, b)
        a, b = (rows[k][rows[k].policy == p].sort_values(w.KEY).reset_index(drop=True) for k in rows)
        cols = ['valid', 'success', 'energy', 'outcome_bin', 'numerical_failure']
        pd.testing.assert_frame_equal(a[w.KEY + cols], b[w.KEY + cols])
        checked[p] = dict(estimates=20 * 21, r2_rows=int((r2['this'].policy == p).sum()), task_rows=len(b))
    contrast = f'this - {other.name}'
    contrasts = []
    for p in (name[p] for p in changed):
        contrasts.append(dict(contrast=contrast, policy=p, endpoint='prediction_R2', trials=20, budget_pa2_s=np.nan,
                              a=r2_vector(r2['this'], p, 20), b=r2_vector(r2['other'], p, 20)))
        for budget in (10., 12.):
            contrasts.append(dict(contrast=contrast, policy=p, endpoint='overturn_success', trials=20,
                                  budget_pa2_s=40 * budget,
                                  a=success_matrix(rows['this'], p, 20, budget), b=success_matrix(rows['other'], p, 20, budget)))
    frame = simultaneous(contrasts, seed=seed) if contrasts else pd.DataFrame(
        columns=['policy', 'endpoint', 'budget_pa2_s', 'value_a', 'value_b', 'difference', 'lower', 'upper',
                 'interval_excludes_zero'])
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    frame.to_csv(comparison / f'versus_{other.name}.csv', index=False)
    record = dict(other=str(other), other_constraint=other_scenario.constraint,
                  other_later_scale=other_scenario.later_scale, identical_policies=checked,
                  changed_policies=[name[p] for p in changed],
                  contrasts=[{k: (None if pd.isna(v) else v) for k, v in row.items()} for row in
                             frame[['policy', 'endpoint', 'budget_pa2_s', 'value_a', 'value_b', 'difference', 'lower',
                                    'upper', 'interval_excludes_zero']].astype(object).to_dict('records')])
    write_json(out / 'provenance' / f'versus_{other.name}.json', record)
    print(json.dumps(record, indent=1, default=float))


def versus_passive(out: Path, scenario: w.Scenario, passive: Path, seed: int = 20261008):
    """Every learned method of this result minus the passive learner of ``passive`` (``passive_e20``).

    The passive learner receives the evidence in all 20 trials and no control input. Contrasts
    pair the training seeds: R2 after trial 20, and success at 400 and 480 pA^2 s after trial 20,
    with one simultaneous interval per endpoint (as ``audit``). Output:
    ``comparison/versus_passive.csv``.
    """
    reference = learned_policies(w.SCENARIOS['passive_e20'])[0]
    r2 = {k: pd.read_csv(d / 'eval' / 'r2_per_session.csv') for k, d in (('this', out), ('passive', passive))}
    rows = {k: task_rows(d / 'eval' / 'task_warm') for k, d in (('this', out), ('passive', passive))}
    contrasts = []
    for p in learned_policies(scenario):
        contrasts.append(dict(contrast='method - passive', policy=p, endpoint='prediction_R2', trials=20,
                              budget_pa2_s=np.nan, a=r2_vector(r2['this'], p, 20),
                              b=r2_vector(r2['passive'], reference, 20)))
        for budget in (10., 12.):
            contrasts.append(dict(contrast='method - passive', policy=p, endpoint='overturn_success', trials=20,
                                  budget_pa2_s=40 * budget, a=success_matrix(rows['this'], p, 20, budget),
                                  b=success_matrix(rows['passive'], reference, 20, budget)))
    frame = simultaneous(contrasts, seed=seed)
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    frame.to_csv(comparison / 'versus_passive.csv', index=False)
    print(frame[['policy', 'endpoint', 'budget_pa2_s', 'value_a', 'value_b', 'difference', 'lower', 'upper',
                 'interval_excludes_zero']].to_string(index=False))


def result(run: Path, out: Path, scenario_name: str, other: Path | None, other_name: str | None,
           identical: list[str] | None = None):
    """One final result folder from one campaign scenario (no simulation)."""
    scenario = w.SCENARIOS[scenario_name]
    assemble_result(run, out, scenario_name)
    protocol(out)
    verify(out, scenario)
    score(out, scenario)
    if 'adaptive' in learned_policies(scenario):  # the passive scenario has no PALDI to audit
        audit(out, scenario)
    evidence(out, scenario)
    if other is not None:
        versus(out, scenario, other, w.SCENARIOS[other_name], identical)


# ------------------------------------------------------------------ evidence-trial diagnostics
def evidence_trials(runs_root: Path, schedule: str, replay: pd.DataFrame, n_trials: int = 10) -> pd.DataFrame:
    """One row per evidence-led trial (1 to ``n_trials``) of every run, with its evidence-only replay.

    Trace states precede each bin's input; the final pre-reset state is not in the trace, so
    post-decision leads cover the recorded bins only (as in the handoff diagnostic).
    """
    replay = replay.set_index(['seed', 'trial'])
    records = []
    for path in sorted(runs_root.glob('*/seed_*/repeat_01')):
        meta = json.loads((path / 'run_metadata.json').read_text())
        policy, seed = ev.evaluated_policy(meta['policy_id']), int(meta['seed'])
        trace = pd.read_csv(path / 'state_action_trace.csv', usecols=['session_index', 'true_z0', 'true_z1',
                                                                      'policy_action_x', 'policy_action_v'])
        for s in meta['session_log'][:n_trials]:
            d = trace[trace.session_index == s['session']]
            assert len(d) == s['bins']
            pool, chosen, bin_ = s['evidence_pool'], s['decision'] - 1, s['decision_bin']
            gap = (d.true_z0 - d.true_z1).to_numpy()
            post = (gap if chosen == 0 else -gap)[bin_:] if chosen >= 0 else np.array([])
            u = d[['policy_action_x', 'policy_action_v']].to_numpy(float)
            base = replay.loc[(seed, s['session'] + 1)]
            assert int(base.pool) == pool, (policy, seed, s)
            base_chosen = int(base.decision) - 1
            records.append(dict(
                schedule=schedule, policy=policy, seed=seed, trial=s['session'] + 1, evidence_pool=pool,
                bins=s['bins'], chosen=chosen, decision_ms=5 * bin_ if chosen >= 0 else np.nan,
                against_evidence=chosen >= 0 and chosen != pool, decision_while_evidence=0 < bin_ <= 200,
                recorded_postdecision_reversal=bool((post < -2).any()),
                recorded_postdecision_lead_changed=bool((post < 0).any()),
                smallest_recorded_postdecision_lead=float(post.min()) if len(post) else np.nan,
                control_rms_pa=float(20 * np.sqrt((u ** 2).sum(1).mean())),
                control_energy_pa2_s=float(400 * 0.005 * (u ** 2).sum()),
                evidence_only_chosen=base_chosen,
                evidence_only_decision_ms=5 * int(base.decision_bin) if base_chosen >= 0 else np.nan,
                choice_differs_from_evidence_only=chosen != base_chosen,
                choice_flipped_vs_evidence_only=chosen >= 0 and base_chosen >= 0 and chosen != base_chosen))
    return pd.DataFrame(records)


def diagnostics(out: Path):
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    replay = pd.concat([pd.read_csv(p) for p in sorted((out / 'evidence_only').glob('group_*.csv'))], ignore_index=True)
    assert len(replay) == 200 and not replay.duplicated(['seed', 'trial']).any()
    trials = pd.concat([evidence_trials(FROZEN_RUNS, 'strong', replay), evidence_trials(weak_runs(out), 'weak', replay)],
                       ignore_index=True)
    assert len(trials) == 2 * 6 * 200
    trials.to_csv(comparison / 'evidence_trials.csv', index=False)
    summary = trials.groupby(['schedule', 'policy'], sort=False).agg(
        trials=('trial', 'size'), decisions=('chosen', lambda c: int((c >= 0).sum())),
        first_choice_against_evidence=('against_evidence', 'sum'),
        decisions_while_evidence=('decision_while_evidence', 'sum'),
        choice_differs_from_evidence_only=('choice_differs_from_evidence_only', 'sum'),
        choice_flipped_vs_evidence_only=('choice_flipped_vs_evidence_only', 'sum'),
        recorded_postdecision_reversals=('recorded_postdecision_reversal', 'sum'),
        recorded_postdecision_lead_changes=('recorded_postdecision_lead_changed', 'sum'),
        smallest_recorded_postdecision_lead=('smallest_recorded_postdecision_lead', 'min'),
        median_decision_ms=('decision_ms', 'median'), mean_trial_bins=('bins', 'mean'),
        control_rms_pa=('control_rms_pa', 'mean'), control_energy_pa2_s=('control_energy_pa2_s', 'mean')).reset_index()
    decided = replay.decision > 0
    base = dict(trials=200, decisions=int(decided.sum()),
                first_choice_against_evidence=int((decided & (replay.decision - 1 != replay.pool)).sum()),
                median_decision_ms=float(np.median(5 * replay.decision_bin[replay.decision > 0])))
    summary.to_csv(comparison / 'evidence_summary.csv', index=False)
    checks = [json.loads(p.read_text()) for p in sorted((out / 'provenance' / 'vm' / 'run_checks').glob('*.json'))]
    bounds = pd.DataFrame([dict(policy=ev.evaluated_policy(c['policy']), seed=c['seed'], geometry=c['geometry'],
                                trials='1-10' if k == 'weak' else '11-20',
                                bound_pa=20 * v['bound'], max_size_pa=20 * v['max_size'], max_ratio=v['max_ratio'],
                                mean_ratio=v['mean_ratio'], at_bound_fraction=v['at_bound_fraction'],
                                identification_bins=sum(c['trial_bins'][:10] if k == 'weak' else c['trial_bins'][10:]))
                           for c in checks for k, v in c['phases'].items()])
    bounds.to_csv(comparison / 'bound_compliance.csv', index=False)
    write_json(comparison / 'evidence_only_summary.json', dict(
        **base, replay_validation=[json.loads(p.read_text())['passed'] for p in sorted((out / 'evidence_only').glob('validation_group_*.json'))],
        note='Zero-control replay of trials 1-10 with the same network seeds; the evidence-only choice of each trial.'))
    print(summary.to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--stage', required=True,
                   choices=['assemble', 'protocol', 'verify', 'score', 'compare', 'diagnostics', 'audit', 'result',
                            'versus_passive', 'all'])
    p.add_argument('--run', type=Path, help='collected gcp_runs/<id> directory (assemble)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--scenario', choices=sorted(w.SCENARIOS), default='weak2_e10', help='audit and result')
    p.add_argument('--compare-to', type=Path, help='result: another result of the same evaluation')
    p.add_argument('--compare-scenario', choices=sorted(w.SCENARIOS), help='result: scenario of --compare-to')
    p.add_argument('--compare-identical', nargs='*', choices=sorted(w.GEOMETRY),
                   help='result: policies that must match --compare-to exactly (default: same catalog entry '
                        'and identification protocol)')
    p.add_argument('--passive', type=Path, help='versus_passive: the passive_e20 result folder')
    a = p.parse_args()
    stages = ['assemble', 'verify', 'score', 'compare', 'diagnostics'] if a.stage == 'all' else [a.stage]
    for stage in stages:
        if stage == 'assemble':
            assemble(a.run, a.out)
        elif stage == 'audit':
            audit(a.out, w.SCENARIOS[a.scenario])
        elif stage == 'result':
            result(a.run, a.out, a.scenario, a.compare_to, a.compare_scenario, a.compare_identical)
        elif stage == 'versus_passive':
            versus_passive(a.out, w.SCENARIOS[a.scenario], a.passive)
        else:
            globals()[stage](a.out)


if __name__ == '__main__':
    main()
