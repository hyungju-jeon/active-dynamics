# TBME Experiments

This directory contains the TBME experiment definitions, TBME-specific catalogs, and TBME figure entrypoints. The experiment execution path is intentionally thin:

1. A suite module declares one manuscript experiment family through `EXPERIMENT_SUITES`.
2. `experiments.tbme.run_tbme_experiments` installs the TBME catalogs and injects those suite definitions.
3. `experiments.run` performs the actual run and summary work.

There is no TBME suite YAML file in the current structure. Suite definitions live next to the experiment they describe.
Suite ids use clean environment slugs such as `duffing` and `gated_duffing`, not old `exp##_` prefixes.

## Directory Layout

- `config/experiment_env.yaml`: TBME environment presets.
- `config/experiment_model.yaml`: TBME policy, objective, and schedule presets.
- `exp_simple_system_identification.py`: base-environment policy comparisons.
- `exp_observation_action_bottleneck.py`: gated-Duffing bottleneck experiments.
- `exp_model_mismatch.py`: nominal model-mismatch experiments.
- `exp_parameter_mismatch_stress.py`: mild and strong parameter-mismatch stress tests.
- `exp_observation_tuning_mismatch.py`: observation-tuning mismatch experiments.
- `exp_objective_ablation.py`: objective ablations.
- `exp_scheduling.py`: planning schedule sweeps.
- `exp_hard_environment.py`: legacy hard-environment suite.
- `exp_observation_mismatch_stress.py`: legacy observation-loading mismatch stress suite.
- `run_tbme_experiments.py`: helper that connects each explicit TBME suite module to `experiments.run`.
- `generate_figures.py`: single entrypoint for TBME figure and asset generation.
- `generate_env_diagnostics.py`: catalog-level dynamics and observation diagnostics.
- `generate_behavior_video.py`: TBME behavior frame/video renderer.
- `tbme_io.py`: shared TBME run trace and metadata-loading helpers.
- `tbme_figures.py`: TBME summary, overview, and experiment-figure implementation.
- `tbme_assets.py`: manuscript asset assembly used by `generate_figures.py assets`.

The shared experiment runtime remains outside this directory:

- `experiments/run.py`
- `experiments/summarize.py`
- `experiments/experiment_definitions.py`
- `actdyn/core/experiment.py`
- `actdyn/utils/plotting.py`

## Catalogs

TBME runs use both the base experiment catalogs and the TBME catalogs:

- environments: `experiments/experiment_env.yaml`, then `experiments/tbme/config/experiment_env.yaml`
- models/policies: `experiments/experiment_model.yaml`, then `experiments/tbme/config/experiment_model.yaml`
- suites: injected from the selected suite module, with file-backed suite catalogs disabled

When the same model or policy id appears in both model catalog files, the later TBME catalog entry overrides the base entry as a whole. This is not a deep merge. Current duplicated ids include `active_myopic`, `active_planning`, and `random`.

## Shared Experiment Families

The default TBME entrypoint is the shared tracks family:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments all --mode all --skip-existing
```

It writes to `results/tbme/tracks/session_<n>` and deduplicates repeated
environment-method pairs in this order:

| Shared group | Source modules |
| --- | --- |
| `simple_system_identification` | `exp_simple_system_identification.py` |
| `observation_action_bottleneck` | `exp_observation_action_bottleneck.py` |
| `model_mismatch` | `exp_model_mismatch.py`, `exp_parameter_mismatch_stress.py`, `exp_observation_tuning_mismatch.py` |
| `objective_ablation` | `exp_objective_ablation.py` |
| `scheduling` | `exp_scheduling.py` |

The default seed counts are manuscript-scale defaults. For smoke tests or debugging, always pass an explicit small `--seeds` value.

## Running Experiments

Run experiment modules from the repository root with `./.venv/bin/python -m`. This keeps the package import path explicit and uses the project environment. If your shell already activates the project environment, plain `python -m` is equivalent.

```bash
./.venv/bin/python -m experiments.tbme.exp_simple_system_identification --mode run --seeds 0 --skip-existing
```

Run all shared tracks, including summaries:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments all \
  --mode all \
  --seeds 0,10,20 \
  --skip-existing
```

Run a small one-policy smoke test:

```bash
./.venv/bin/python -m experiments.tbme.exp_simple_system_identification \
  --mode run \
  --exp-ids duffing \
  --policy-ids random \
  --seeds 0 \
  --total-steps 1 \
  --base-dir /tmp/tbme_smoke \
  --skip-existing
```

The helper can also run a family by module name:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments exp_objective_ablation \
  --mode summary \
  --exp-ids gated_duffing \
  --seeds 0
```

Do not prefer direct file execution such as `./.venv/bin/python experiments/tbme/exp_simple_system_identification.py`. The current entrypoints use package imports and are meant to be run with `./.venv/bin/python -m` from the repository root.

## Common Arguments

Each suite module accepts these TBME-level defaults and forwards all other arguments to `experiments.run`:

- `--exp-ids`: comma-separated suite ids; defaults to that experiment family's suites.
- `--base-dir`: output root; defaults to `results/tbme/tracks`.
- `--seeds`: comma-separated integer seeds; defaults to the family seed range.

Common forwarded arguments include:

- `--mode {run,summary,all}`
- `--policy-ids`
- `--repeats`
- `--skip-existing`
- `--total-steps`
- `--q-theta`
- `--parameter-prior-covariance`
- `--eig-gamma`

Use `--help` on the TBME entrypoints to inspect the current parser. Prefer these entrypoints because they install the TBME catalog stack before calling the generic runner:

```bash
./.venv/bin/python -m experiments.tbme.exp_simple_system_identification --help
./.venv/bin/python -m experiments.tbme.run_tbme_experiments --help
```

## Outputs

Runs write one session under the selected `--base-dir`. A typical run contains:

- `session_metadata.json`: resolved catalogs, command line, experiments, policies, seeds, and run summary.
- `experiment_driver.log`: redirected stdout and stderr when running non-interactively.
- `<env>/<policy_id>/seed_<seed>/repeat_<repeat>/run_metadata.json`: per-run metadata.
- Trace CSV files such as `parameter_error_trace.csv`, `trajectory_r2_trace.csv`, `embedding_estimate_trace.csv`, `information_trace.csv`, and `state_action_trace.csv`.
- Summary artifacts when `--mode summary` or `--mode all` is used.

## Figure Generation

Use `generate_figures.py` as the single figure entrypoint:

```bash
./.venv/bin/python -m experiments.tbme.generate_figures --help
./.venv/bin/python -m experiments.tbme.generate_figures summary
./.venv/bin/python -m experiments.tbme.generate_figures overview
./.venv/bin/python -m experiments.tbme.generate_figures experiment
./.venv/bin/python -m experiments.tbme.generate_figures assets
./.venv/bin/python -m experiments.tbme.generate_figures diagnostics
./.venv/bin/python -m experiments.tbme.generate_figures all
./.venv/bin/python -m experiments.tbme.tbme_assets --help
```

The figure code keeps TBME result-group definitions in `tbme_figures.py`. If result roots are renamed, update the `GROUPS` table there before relying on the figure commands.
The diagnostics command does not require completed runs; by default it plots every unique environment preset used by the shared TBME suites.

## Reproducibility Checklist

Before launching a manuscript-scale run, record:

- the exact module command;
- `--exp-ids`, if not using the family default;
- `--policy-ids`, if filtering policies;
- `--seeds` and `--repeats`;
- `--base-dir`;
- whether `--total-steps` overrides the suite default;
- the active git commit written to `session_metadata.json`.

For quick validation, prefer one suite, one policy, one seed, and `--total-steps 1` before launching the full seed grid.
