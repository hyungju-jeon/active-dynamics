"""Regenerate reviewed manuscript layouts in a separate experiment output tree.

Run from the repository root with ``python -m experiments.tbme.figures.session``.
Affected policies come exclusively from the corrected session; all other policies
come from the explicitly named reference. Neither source tree is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import assets as a, groups

WIDTH = 516.0 / 72.27


def prepare_inputs(corrected: Path, reference: Path, output: Path) -> Path:
    """Build a figure-only view and aggregate saved scores without reevaluation."""
    view = output / 'inputs' / 'session_1'
    provenance = []
    nonfinite = []
    for env in sorted((reference / 'tracks').iterdir()):
        if not env.is_dir():
            continue
        target = view / 'tracks' / env.name
        summary = target / 'summary'
        summary.mkdir(parents=True, exist_ok=True)
        updated = corrected / 'tracks' / env.name
        changed = {p.name for p in updated.glob('*') if list(p.glob('seed_*'))}
        for policy in sorted(env.iterdir()):
            if not list(policy.glob('seed_*')):
                continue
            source = updated / policy.name if policy.name in changed else policy
            metas = sorted(source.glob('seed_*/repeat_01/run_metadata.json'))
            assert len(metas) == 100, (source, len(metas))
            assert {json.loads(p.read_text())['seed'] for p in metas} == set(range(100))
            assert all(json.loads(p.read_text())['status'] == 'completed' for p in metas)
            link = target / policy.name
            if link.is_symlink():
                assert link.resolve() == source.resolve()
            else:
                assert not link.exists()
                link.symlink_to(source.resolve(), target_is_directory=True)
            provenance.append({'env': env.name, 'policy': policy.name,
                               'source': str(source), 'corrected': policy.name in changed,
                               'n_seeds': len(metas)})
        metrics = pd.read_csv(env / 'summary' / 'metrics.csv')
        curves = pd.read_csv(env / 'summary' / 'trajectory_r2_over_steps.csv')
        if changed:
            fresh = pd.read_csv(updated / 'summary' / 'metrics.csv')
            metrics = pd.concat([metrics[~metrics.policy_id.isin(changed)], fresh], ignore_index=True)
            new_curves = []
            for policy in sorted(changed):
                traces = []
                for path in sorted((updated / policy).glob('seed_*/repeat_01/trajectory_r2_trace.csv')):
                    t = pd.read_csv(path)
                    assert not t.step.duplicated().any()
                    traces.append(t.set_index('step').trajectory_r2)
                values = pd.concat(traces, axis=1)
                assert values.shape[1] == 100
                assert all(t.index.equals(traces[0].index) for t in traces)
                count = int((~np.isfinite(values.to_numpy())).sum())
                if count:
                    nonfinite.append({'env': env.name, 'policy': policy, 'points': count})
                    # These mismatch experiments are outside the manuscript panels.
                    # Keep NaNs in their summaries; do not silently discard samples.
                    assert env.name not in {'duffing', 'damped_pendulum', 'gated_duffing',
                        'gated_duffing_asymmetric', 'gated_duffing_challenging',
                        'gated_duffing_observation_bottleneck_mild',
                        'gated_duffing_observation_bottleneck_strong', 'three_gate_diagnostic'}
                for step, row in values.iterrows():
                    v = row.to_numpy(dtype=float)
                    new_curves.append({'policy_id': policy, 'step': int(step),
                        'trajectory_r2_mean': v.mean(), 'value_mean': v.mean(),
                        'value_sem': v.std(ddof=1) / np.sqrt(len(v)),
                        'value_median': np.median(v), 'value_q25': np.quantile(v, .25),
                        'value_q75': np.quantile(v, .75), 'n_points': len(v)})
            curves = pd.concat([curves[~curves.policy_id.isin(changed)], pd.DataFrame(new_curves)], ignore_index=True)
        assert not metrics.duplicated(['policy_id', 'seed']).any()
        assert not curves.duplicated(['policy_id', 'step']).any()
        metrics.to_csv(summary / 'metrics.csv', index=False)
        curves.sort_values(['policy_id', 'step']).to_csv(summary / 'trajectory_r2_over_steps.csv', index=False)
    (output / 'nonfinite_trace_points.json').write_text(json.dumps(nonfinite, indent=2))
    (output / 'data_sources.json').write_text(json.dumps(provenance, indent=2))
    assert sum(x['corrected'] for x in provenance) == 96
    return view


def bar_layout(path: Path, panels: list[list[a._ExperimentSuiteSource]],
               policies: list[str], mode: str, *, compact: bool = False) -> None:
    """Match the edited two-panel constraints and compact objective figures."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    a._apply_asset_style(plt)
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDTH / 2 if compact else WIDTH, 2.65 if compact else 1.95), squeeze=False)
    records = []
    for i, sources in enumerate(panels):
        rows = a._asset_method_metric_rows(sources, policies, r2_summary=mode)
        assert all(r['n_r2'] == 100 and r['n_r2_nonfinite'] == 0 for r in rows)
        records.extend(rows)
        ax = axes[0, i]
        a._asset_plot_final_bar(path, sources=sources, policy_ids=policies,
                               metric_rows=rows, single_column=compact, ax=ax)
        if len(panels) > 1:
            ax.set_title(chr(65+i), loc='left', fontweight='bold', fontsize=10, pad=3)
        if i:
            ax.set_ylabel('')
    fig.legend([Line2D([0], [0], color=a._asset_baseline_policy_color(p), lw=1.6) for p in policies],
               [a._asset_policy_label(p) for p in policies], loc='upper center',
               bbox_to_anchor=(.5, .995), ncol=4 if compact else len(policies),
               fontsize=6, handlelength=1.6, columnspacing=.9)
    fig.tight_layout(rect=(0, 0, 1, .83 if compact else .91), w_pad=1.2)
    a._asset_write_method_csv(path.with_suffix('.csv'), records, r2_summary=mode)
    a.save_figure(fig, path, plt_module=plt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, default=Path('results/tbme/time_alignment/session_1'))
    parser.add_argument('--reference', type=Path, default=Path('results/tbme/session_4'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/tbme/time_alignment/session_1/figures_review'))
    args = parser.parse_args()
    corrected, reference, output = (p.resolve() for p in (args.session, args.reference, args.output_dir))
    manuscript = groups.REPO_ROOT / 'docs/active-dynamics-writing'
    assert not output.is_relative_to(manuscript)
    assert not output.is_relative_to(reference)
    output.mkdir(parents=True, exist_ok=True)
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in manuscript.rglob('*') if p.is_file() and p.suffix in {'.pdf', '.tex', '.svg'}}
    (output / 'manuscript_before.json').write_text(json.dumps(protected, indent=2))
    view = prepare_inputs(corrected, reference, output)
    groups.set_results_dir(view.parent)
    source = lambda env, label: a._ExperimentSuiteSource(env, label, view/'tracks'/env)
    panels = [[source('gated_duffing', 'Default'),
               source('gated_duffing_observation_bottleneck_mild', 'SNR -10'),
               source('gated_duffing_observation_bottleneck_strong', 'SNR -15')],
              [source('gated_duffing', 'Default'), source('gated_duffing_asymmetric', 'Biased')]]
    objectives = [[source('gated_duffing', 'Default'), source('gated_duffing_asymmetric', 'Asymmetric'),
                   source('gated_duffing_challenging', 'Challenging')]]
    for mode, prefix in [('median_iqr', 'figure'), ('mean_sem', 'appendix')]:
        baseline = 'figure_active_baseline' if prefix == 'figure' else 'appendix_active_vs_baselines'
        a._asset_plot_active_vs_baselines(output/(baseline+'.pdf'), r2_summary=mode)
        bar_layout(output/(prefix+'_constraints.pdf'), panels, a._ASSET_MATCHED_POLICIES, mode)
        bar_layout(output/(prefix+'_objective_ablation.pdf'), objectives, a._experiment_OBJECTIVE_POLICIES, mode, compact=True)
        print('Rendered', mode, flush=True)
    gate_roots = [p.resolve() for p in (view/'tracks/three_gate_diagnostic').iterdir() if p.is_symlink()]
    a._asset_plot_gate_diagnostic(output/'figure_gate_diagnostic.pdf', r2_summary='median_iqr', result_roots=gate_roots)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in protected.items())
    (output/'generation.json').write_text(json.dumps({'corrected': str(corrected), 'reference': str(reference),
        'manuscript_unchanged': True, 'command': 'python -m experiments.tbme.figures.session',
        'n_corrected_runs': 9600, 'intervals': {'figure': 'median and IQR', 'appendix': 'mean and SEM'},
        'smoothing': 'none', 'curve_axis_floor': .25, 'bar_axis_floor': 0,
        'baseline_reuse': 'See data_sources.json; unchanged policies come from session_4.'}, indent=2))
    print(output, flush=True)


if __name__ == '__main__':
    main()
