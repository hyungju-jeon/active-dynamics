"""Compare the SNN identification scenarios (no simulation).

Five conditions of the Native+Uniform protocol share the 20 training seeds, calibrated readouts,
priors, network seeds of each trial, and the task panel:

* ``strong``: the selected manuscript result: 10 evidence trials and 10 exploration trials, 20 pA
  throughout (``results/tnsre/20261007_snn_native_uniform``).
* ``weak2_e10``: 2 pA in evidence trials 1-10, then 20 pA (``results/archive/experiments/
  20261007_snn_weak_evidence``).
* ``weak10_e10``: 10 pA in evidence trials 1-10, then 20 pA.
* ``weak2_e5``: 2 pA in evidence trials 1-5, then 15 exploration trials at 20 pA.
* ``passive_e20``: 20 evidence trials without control; the learner's only input is the evidence.

The last three come from one GCP campaign of ``experiments.tnsre.snn_weak_evidence``; each is
assembled and verified like the 2-pA run (``snn_weak_evidence_finalize``). Stages:

* ``assemble``: per-scenario folders ``<out>/scenarios/<name>`` and the campaign provenance.
* ``verify``: per-scenario checks, plus identical evidence-only replays across scenarios and
  campaigns, passive runs equal to their evidence-only replay, and ``weak2_e5`` models through
  trial 5 equal to the 2-pA run's (same trials, separate campaign).
* ``score``: per-scenario summaries and main tables.
* ``compare``: descriptive table and paired training-seed contrasts (``comparison/``).
* ``diagnostics``: evidence-trial choices, input size, decision times, identification time, bounds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from experiments.tnsre import snn_weak_evidence as w
from experiments.tnsre import snn_weak_evidence_finalize as f
from experiments.tnsre.wong_wang_full import write_json

WEAK2 = Path('results/archive/experiments/20261007_snn_weak_evidence')
NEW = ('weak10_e10', 'weak2_e5', 'passive_e20')
CONDITIONS = ('strong', 'weak2_e10', 'weak10_e10', 'weak2_e5', 'passive_e20')
LABELS = {'strong': '20 pA x 10 evidence + 10 exploration (selected)',
          'weak2_e10': '2 pA x 10 evidence + 10 exploration',
          'weak10_e10': '10 pA x 10 evidence + 10 exploration',
          'weak2_e5': '2 pA x 5 evidence + 15 exploration',
          'passive_e20': '20 evidence trials, no control'}
EVIDENCE_TRIALS = {'strong': 10, **{name: w.SCENARIOS[name].evidence_trials for name in w.SCENARIOS}}
PASSIVE = 'off_policy'
BOOTSTRAP_SEED = 20261008
REPLAY_KEYS = ['pool', 'decision', 'decision_bin', 'bins']


def folder(out: Path, condition: str) -> Path:
    return {'strong': f.FROZEN, 'weak2_e10': WEAK2}.get(condition, out / 'scenarios' / condition)


def runs_root(out: Path, condition: str) -> Path:
    if condition == 'strong':
        return f.FROZEN_RUNS
    return w.runs_root(folder(out, condition), w.SCENARIOS[condition])


def learned(condition: str) -> tuple[str, ...]:
    return f.LEARNED if condition == 'strong' else f.learned_policies(w.SCENARIOS[condition])


# ------------------------------------------------------------------ assemble
def assemble(run: Path, out: Path):
    """Per-scenario result folders from the collected campaign ``gcp_runs/<id>``, plus its provenance."""
    campaign = run / 'results' / 'evidence_scenarios'
    for name in NEW:
        f.assemble(run, out / 'scenarios' / name, source=campaign / name)
    vm = out / 'provenance' / 'vm'
    vm.mkdir(parents=True)
    shutil.copytree(campaign / 'logs', vm / 'logs')
    for path in sorted(campaign.glob('*_group_*.json')):
        shutil.copy2(path, vm / path.name)
    shutil.copy2(run / 'commit.txt', out / 'provenance' / 'commit.txt')
    shutil.copytree(run / 'logs', out / 'provenance' / 'cloud_logs')
    references = {'strong': [f.FROZEN / 'provenance' / 'training_verification.json', f.FROZEN / 'main_table.csv'],
                  'weak2_e10': [WEAK2 / 'provenance' / 'verification.json', WEAK2 / 'main_table.csv',
                                WEAK2 / 'provenance' / 'commit.txt']}
    write_json(out / 'provenance' / 'inputs.json', dict(
        source_run=str(run), reference_folders={k: str(folder(out, k)) for k in ('strong', 'weak2_e10')},
        reference_files={str(p): f.sha256(p) for paths in references.values() for p in paths}))


# ------------------------------------------------------------------ verify
def replays(out: Path) -> dict[str, pd.DataFrame]:
    """Evidence-only replay of each campaign scenario and of the 2-pA run, indexed by (seed, trial)."""
    sources = {'weak2_e10': WEAK2, **{name: out / 'scenarios' / name for name in NEW}}
    return {name: pd.concat([pd.read_csv(p) for p in sorted((path / 'evidence_only').glob('group_*.csv'))],
                            ignore_index=True).set_index(['seed', 'trial']).sort_index()
            for name, path in sources.items()}


def evidence_only(out: Path) -> pd.DataFrame:
    """The evidence-only choice of trials 1-20 for every seed (the passive scenario replays all 20)."""
    return replays(out)['passive_e20'].reset_index()


def shared_prefix(out: Path) -> dict:
    """``weak2_e5`` repeats trials 1-5 of the 2-pA run: its models up to trial 5 must equal that run's."""
    keys = ['policy', 'seed', 'sessions']
    a, b = (pd.read_csv(folder(out, c) / 'eval' / 'r2_per_session.csv') for c in ('weak2_e10', 'weak2_e5'))
    r2 = a[a.sessions <= 5].merge(b[b.sessions <= 5], on=keys, suffixes=('_a', '_b'), validate='one_to_one')
    assert len(r2) == 6 * 20 * 6 and (r2.r2_a == r2.r2_b).all() and (r2.theta_a == r2.theta_b).all()
    keys = ['controller', 'policy', 'seed', 'checkpoint', 'task', 'budget', 'task_seed']
    a, b = (f.task_rows(folder(out, c) / 'eval' / 'task_warm') for c in ('weak2_e10', 'weak2_e5'))
    rows = a[a.checkpoint.isin([1, 5])].merge(b[b.checkpoint.isin([1, 5])], on=keys, suffixes=('_a', '_b'),
                                              validate='one_to_one')
    assert len(rows) == 2 * 6 * 20 * 8  # checkpoints x policies x seeds x task seeds (480 pA^2 s only)
    for column in ('valid', 'success', 'energy', 'outcome_bin', 'numerical_failure'):
        assert rows[column + '_a'].equals(rows[column + '_b']), column
    return dict(r2_rows_trials_0_to_5=len(r2), task_rows_trials_1_and_5=len(rows))


def verify(out: Path):
    """Each new scenario passes the 2-pA run's checks; replays and passive runs agree exactly."""
    records = {}
    for name in NEW:
        f.verify(out / 'scenarios' / name, w.SCENARIOS[name])
        records[name] = json.loads((out / 'scenarios' / name / 'provenance' / 'verification.json').read_text())
    # A trial's evidence-only outcome depends only on its network seed: replays from different
    # scenarios and campaigns must agree on every shared (seed, trial).
    replay = replays(out)
    full = replay['passive_e20']
    assert len(full) == 20 * 20 and set(full.index.get_level_values('seed')) == set(w.SEEDS)
    agreement = {}
    for name, frame in replay.items():
        assert len(frame) == 20 * EVIDENCE_TRIALS[name] and frame.index.is_unique
        pd.testing.assert_frame_equal(frame[REPLAY_KEYS], full.loc[frame.index, REPLAY_KEYS])
        agreement[name] = len(frame)
    # The passive identification run is its own evidence-only replay.
    passive = []
    for path in sorted(runs_root(out, 'passive_e20').glob(f'{PASSIVE}/seed_*/repeat_01/run_metadata.json')):
        meta = json.loads(path.read_text())
        log = pd.DataFrame(meta['session_log']).assign(seed=int(meta['seed']), trial=lambda d: d.session + 1)
        log = log.rename(columns={'evidence_pool': 'pool'}).set_index(['seed', 'trial'])
        np.testing.assert_array_equal(log[REPLAY_KEYS].to_numpy(), full.loc[log.index, REPLAY_KEYS].to_numpy())
        passive.append(int(meta['seed']))
    assert sorted(passive) == w.SEEDS
    record = dict(passed=True, scenarios={k: dict(passed=v['passed'], runs=v['runs'], run_checks=v['run_checks'],
                                                  task_rows=v['task_rows'], learned_rows=v['learned_rows'],
                                                  numerical_failures=v['numerical_failures'],
                                                  session_rule_fields_changed=v['session_rule_fields_changed'])
                                              for k, v in records.items()},
                  replay_rows_matching_the_20_trial_replay=agreement,
                  passive_runs_equal_their_evidence_only_replay=len(passive),
                  weak2_e5_equals_weak2_e10_through_trial_5=shared_prefix(out))
    write_json(out / 'provenance' / 'verification.json', record)
    print(json.dumps(record, indent=1))


def score(out: Path):
    for name in NEW:
        f.score(out / 'scenarios' / name, w.SCENARIOS[name])


# ------------------------------------------------------------------ compare
def load(out: Path) -> tuple[dict, dict]:
    r2 = {c: pd.read_csv(folder(out, c) / 'eval' / 'r2_per_session.csv') for c in CONDITIONS}
    rows = {c: f.task_rows(folder(out, c) / 'eval' / 'task_warm') for c in CONDITIONS}
    return r2, rows


def compare(out: Path):
    """Descriptive table and three families of paired contrasts (one simultaneous interval per endpoint).

    * ``between_scenarios``: per method, each new schedule against the 2-pA run and every
      schedule against the selected one.
    * ``versus_passive``: per method and active condition, method minus the passive learner.
    * ``paldi_minus_method``: within each active condition, PALDI minus every other method.

    R2 after trial 20 and success at 480 pA^2 s after trial 20; the paired bootstrap resamples
    the 20 training seeds (``snn_weak_evidence_finalize.simultaneous``).
    """
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    r2, rows = load(out)
    active = CONDITIONS[:-1]

    def pair(contrast, family, policy_a, a, policy_b, b, endpoint):
        if endpoint == 'prediction_R2':
            va, vb = f.r2_vector(r2[a], policy_a, 20), f.r2_vector(r2[b], policy_b, 20)
            budget = np.nan
        else:
            va, vb = f.success_matrix(rows[a], policy_a, 20, 12.), f.success_matrix(rows[b], policy_b, 20, 12.)
            budget = 480.
        return dict(family=family, contrast=contrast, condition_a=a, policy_a=policy_a, condition_b=b,
                    policy_b=policy_b, endpoint=endpoint, trials=20, budget_pa2_s=budget, a=va, b=vb)

    families = {'between_scenarios': [], 'versus_passive': [], 'paldi_minus_method': []}
    for endpoint in ('prediction_R2', 'overturn_success'):
        for policy in f.LEARNED:
            for a, b in (('weak10_e10', 'weak2_e10'), ('weak2_e5', 'weak2_e10'), ('weak2_e10', 'strong'),
                         ('weak10_e10', 'strong'), ('weak2_e5', 'strong')):
                families['between_scenarios'].append(pair(f'{a} - {b}', 'between_scenarios', policy, a, policy, b, endpoint))
            for a in active:
                families['versus_passive'].append(pair(f'{a} - passive_e20', 'versus_passive', policy, a, PASSIVE,
                                                       'passive_e20', endpoint))
        for a in active:
            for policy in f.LEARNED[1:]:
                families['paldi_minus_method'].append(pair(f'PALDI - method ({a})', 'paldi_minus_method', 'adaptive', a,
                                                           policy, a, endpoint))
    frames = [f.simultaneous(contrasts, seed=BOOTSTRAP_SEED) for contrasts in families.values()]
    pd.concat(frames, ignore_index=True).to_csv(comparison / 'contrasts.csv', index=False)
    # Descriptive: prediction quartiles and mean success over training seeds and valid task trials.
    table = []
    for condition in CONDITIONS:
        for policy in learned(condition):
            record = dict(condition=condition, label=LABELS[condition], policy=policy)
            for k in (5, 10, 20):
                q = np.percentile(f.r2_vector(r2[condition], policy, k), [25, 50, 75])
                record.update({f'r2_t{k}_q25': q[0], f'r2_t{k}_median': q[1], f'r2_t{k}_q75': q[2]})
            for k in (1, 5, 10, 15):
                record[f'success_480_t{k}'] = float(f.success_matrix(rows[condition], policy, k, 12.).mean())
            for budget in (5., 10., 12., 15.):
                record[f'success_{40 * budget:g}_t20'] = float(f.success_matrix(rows[condition], policy, 20, budget).mean())
            table.append(record)
    pd.DataFrame(table).to_csv(comparison / 'summary.csv', index=False)
    print(pd.DataFrame(table)[['condition', 'policy', 'r2_t10_median', 'r2_t20_median', 'success_480_t10',
                               'success_480_t20']].to_string())


# ------------------------------------------------------------------ evidence-trial diagnostics
def identification_time(out: Path) -> pd.DataFrame:
    """Total identification time (s) and evidence-trial time of every run (5-ms bins)."""
    records = []
    for condition in CONDITIONS:
        for path in sorted(runs_root(out, condition).glob('*/seed_*/repeat_01/run_metadata.json')):
            meta = json.loads(path.read_text())
            bins = np.array([s['bins'] for s in meta['session_log']])
            n = EVIDENCE_TRIALS[condition]
            records.append(dict(condition=condition, policy=f.ev.evaluated_policy(meta['policy_id']),
                                seed=int(meta['seed']), identification_s=0.005 * bins.sum(),
                                evidence_trials_s=0.005 * bins[:n].sum()))
    return pd.DataFrame(records)


def diagnostics(out: Path):
    comparison = out / 'comparison'; comparison.mkdir(exist_ok=True)
    replay = evidence_only(out)
    trials = pd.concat([f.evidence_trials(runs_root(out, c), c, replay, EVIDENCE_TRIALS[c]) for c in CONDITIONS],
                       ignore_index=True).rename(columns={'schedule': 'condition'})
    sizes = trials.groupby(['condition', 'policy']).size()
    assert all(sizes[c].eq(20 * EVIDENCE_TRIALS[c]).all() for c in CONDITIONS)
    trials.to_csv(comparison / 'evidence_trials.csv', index=False)
    summary = trials.groupby(['condition', 'policy'], sort=False).agg(
        trials=('trial', 'size'), decisions=('chosen', lambda c: int((c >= 0).sum())),
        first_choice_against_evidence=('against_evidence', 'sum'),
        choice_flipped_vs_evidence_only=('choice_flipped_vs_evidence_only', 'sum'),
        recorded_postdecision_reversals=('recorded_postdecision_reversal', 'sum'),
        median_decision_ms=('decision_ms', 'median'), evidence_only_median_decision_ms=('evidence_only_decision_ms', 'median'),
        mean_trial_bins=('bins', 'mean'), control_rms_pa=('control_rms_pa', 'mean')).reset_index()
    time = identification_time(out).groupby(['condition', 'policy'], sort=False)[
        ['identification_s', 'evidence_trials_s']].mean().reset_index()
    summary = summary.merge(time, on=['condition', 'policy'], how='left', validate='one_to_one')
    summary.to_csv(comparison / 'evidence_summary.csv', index=False)
    checks = [(name, json.loads(p.read_text())) for name in NEW
              for p in sorted((out / 'scenarios' / name / 'provenance' / 'vm' / 'run_checks').glob('*.json'))]
    bounds = pd.DataFrame([dict(condition=name, policy=f.ev.evaluated_policy(c['policy']), seed=c['seed'],
                                geometry=c['geometry'], trials=v['trials'], bound_pa=20 * v['bound'],
                                max_size_pa=20 * v['max_size'], max_ratio=v['max_ratio'], mean_ratio=v['mean_ratio'],
                                at_bound_fraction=v['at_bound_fraction'])
                           for name, c in checks for v in c['phases'].values()])
    bounds.to_csv(comparison / 'bound_compliance.csv', index=False)
    print(summary.to_string())
    print(bounds.groupby(['condition', 'trials', 'geometry']).agg(bound=('bound_pa', 'first'),
                                                                   max_pa=('max_size_pa', 'max')).to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--stage', required=True, choices=['assemble', 'verify', 'score', 'compare', 'diagnostics', 'all'])
    p.add_argument('--run', type=Path, help='collected gcp_runs/<id> directory (assemble)')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    stages = ['assemble', 'verify', 'score', 'compare', 'diagnostics'] if a.stage == 'all' else [a.stage]
    for stage in stages:
        if stage == 'assemble':
            assemble(a.run, a.out)
        else:
            globals()[stage](a.out)


if __name__ == '__main__':
    main()
