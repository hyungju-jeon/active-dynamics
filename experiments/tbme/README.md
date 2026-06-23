# TBME Experiments

This directory contains the TBME experiment definitions, TBME-specific catalogs, and TBME figure entrypoints. The experiment execution path is intentionally thin:

1. An `exp*.py` file declares one manuscript experiment family through `EXPERIMENT_SUITES`.
2. `experiments.tbme.run_tbme_experiments` installs the TBME catalogs and injects those suite definitions.
3. `experiments.run` performs the actual run and summary work.

There is no TBME suite YAML file in the current structure. Suite definitions live next to the experiment they describe.

## Directory Layout

- `config/experiment_env.yaml`: TBME environment presets.
- `config/experiment_model.yaml`: TBME policy, objective, and schedule presets.
- `exp01_baseEnv.py`: base-environment policy comparisons.
- `exp02_hardEnv.py`: harder environment policy comparisons.
- `exp03_schedule.py`: planning schedule sweeps.
- `exp04_mismatch.py`: nominal model-mismatch experiments.
- `exp05_ablation.py`: objective ablations.
- `exp06_bottleneck.py`: asymmetric-basin bottleneck experiments.
- `exp07_mismatch_stress.py`: mild and strong observation-loading mismatch stress tests.
- `exp08_parameter_mismatch_stress.py`: mild and strong parameter-mismatch stress tests.
- `run_tbme_experiments.py`: helper that connects each explicit TBME suite module to `experiments.run`.
- `generate_figures.py`: single entrypoint for TBME figure and asset generation.
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
- suites: injected from the selected `exp*.py` module, with file-backed suite catalogs disabled

When the same model or policy id appears in both model catalog files, the later TBME catalog entry overrides the base entry as a whole. This is not a deep merge. Current duplicated ids include `active_myopic`, `active_planning`, and `random`.

## Experiment Families

| Module | Default output root | Default seeds | Suite ids |
| --- | --- | ---: | --- |
| `exp01_baseEnv.py` | `results/tbme/exp01_base` | 1000 | `exp01_duffing`, `exp01_damped_pendulum`, `exp01_asymmetric_basin` |
| `exp02_hardEnv.py` | `results/tbme/exp02_hard` | 1000 | `exp02_hard_duffing`, `exp02_hard_asymmetric_basin`, `exp02_hard_damped_pendulum` |
| `exp03_schedule.py` | `results/tbme/exp03_schedule` | 1000 | `exp03_schedule_duffing`, `exp03_schedule_damped_pendulum`, `exp03_schedule_asymmetric_basin` |
| `exp04_mismatch.py` | `results/tbme/exp04_mismatch` | 1000 | `exp04_duffing_parameter_mismatch`, `exp04_asymmetric_basin_parameter_mismatch` |
| `exp05_ablation.py` | `results/tbme/exp05_ablation` | 100 | `exp05_asymmetric_basin_objective_ablation`, `exp05_hard_asymmetric_basin_objective_ablation` |
| `exp06_bottleneck.py` | `results/tbme/exp06_bottleneck` | 500 | `exp06_asymmetric_basin_observation_bottleneck_mild`, `exp06_asymmetric_basin_observation_bottleneck_strong`, `exp06_asymmetric_basin_action_bottleneck_mild`, `exp06_asymmetric_basin_action_bottleneck_strong` |
| `exp07_mismatch_stress.py` | `results/tbme/exp07_mismatch_stress` | 100 | `exp07_duffing_observation_mismatch_mild`, `exp07_duffing_observation_mismatch_strong`, `exp07_asymmetric_basin_observation_mismatch_mild`, `exp07_asymmetric_basin_observation_mismatch_strong` |
| `exp08_parameter_mismatch_stress.py` | `results/tbme/exp08_parameter_mismatch_stress` | 100 | `exp08_duffing_parameter_mismatch_mild`, `exp08_duffing_parameter_mismatch_strong`, `exp08_asymmetric_basin_parameter_mismatch_mild`, `exp08_asymmetric_basin_parameter_mismatch_strong` |

The default seed counts are manuscript-scale defaults. For smoke tests or debugging, always pass an explicit small `--seeds` value.

## Running Experiments

Run experiment modules from the repository root with `./.venv/bin/python -m`. This keeps the package import path explicit and uses the project environment. If your shell already activates the project environment, plain `python -m` is equivalent.

```bash
./.venv/bin/python -m experiments.tbme.exp01_baseEnv --mode run --seeds 0 --skip-existing
```

Run a complete family, including summaries and videos:

```bash
./.venv/bin/python -m experiments.tbme.exp01_baseEnv --mode all --seeds 0,10,20 --skip-existing
```

Run a small one-policy smoke test:

```bash
./.venv/bin/python -m experiments.tbme.exp01_baseEnv \
  --mode run \
  --exp-ids exp01_duffing \
  --policy-ids random \
  --seeds 0 \
  --total-steps 1 \
  --base-dir /tmp/tbme_smoke_exp01 \
  --skip-existing
```

The helper can also run a family by module name:

```bash
./.venv/bin/python -m experiments.tbme.run_tbme_experiments exp05_ablation \
  --mode summary \
  --exp-ids exp05_asymmetric_basin_objective_ablation \
  --seeds 0
```

Do not prefer direct file execution such as `./.venv/bin/python experiments/tbme/exp01_baseEnv.py`. The current entrypoints use package imports and are meant to be run with `./.venv/bin/python -m` from the repository root.

## Common Arguments

Each `exp*.py` module accepts these TBME-level defaults and forwards all other arguments to `experiments.run`:

- `--exp-ids`: comma-separated suite ids; defaults to that experiment family's suites.
- `--base-dir`: output root; defaults to the family root shown above.
- `--seeds`: comma-separated integer seeds; defaults to the family seed range.

Common forwarded arguments include:

- `--mode {run,summary,video,all}`
- `--policy-ids`
- `--repeats`
- `--skip-existing`
- `--total-steps`
- `--q-theta`
- `--parameter-prior-covariance`
- `--eig-gamma`
- `--stride`, `--fps`, and `--grid-lim` for videos

Use `--help` on the TBME entrypoints to inspect the current parser. Prefer these entrypoints because they install the TBME catalog stack before calling the generic runner:

```bash
./.venv/bin/python -m experiments.tbme.exp01_baseEnv --help
./.venv/bin/python -m experiments.tbme.run_tbme_experiments --help
```

## Outputs

Runs write one session under the selected `--base-dir`. A typical run contains:

- `session_metadata.json`: resolved catalogs, command line, experiments, policies, seeds, and run summary.
- `experiment_driver.log`: redirected stdout and stderr when running non-interactively.
- `<suite_id>/track/<policy_id>/seed_<seed>/repeat_<repeat>/run_metadata.json`: per-run metadata.
- Trace CSV files such as `parameter_error_trace.csv`, `trajectory_r2_trace.csv`, `embedding_estimate_trace.csv`, `information_trace.csv`, and `state_action_trace.csv`.
- Summary and video artifacts when `--mode summary`, `--mode video`, or `--mode all` is used.

## Figure Generation

Use `generate_figures.py` as the single figure entrypoint:

```bash
./.venv/bin/python -m experiments.tbme.generate_figures --help
./.venv/bin/python -m experiments.tbme.generate_figures summary
./.venv/bin/python -m experiments.tbme.generate_figures overview
./.venv/bin/python -m experiments.tbme.generate_figures experiment
./.venv/bin/python -m experiments.tbme.generate_figures assets
./.venv/bin/python -m experiments.tbme.generate_figures all
./.venv/bin/python -m experiments.tbme.tbme_assets --help
```

The figure code keeps TBME result-group definitions in `tbme_figures.py`. If result roots are renamed, update the `GROUPS` table there before relying on the figure commands.

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
