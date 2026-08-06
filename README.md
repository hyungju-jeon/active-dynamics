# Active Dynamics

Active learning framework for latent dynamical system identification.

## Installation

```bash
uv sync
```

## Supported Experiment Tracks

- `experiments/active_embedding`
- `experiments/ciss`
- `experiments/_hydra_templates`

## Single Entry Point

Use the package CLI as the standard runner:

```bash
uv run actdyn --help
uv run actdyn run --config experiments/active_embedding/conf/config.yaml
uv run actdyn sweep --config-path experiments/ciss/conf
uv run actdyn analyze results
uv run actdyn analyze results --summary --save-summary
```

Shared experiment modules are named by role:

- `experiments/run.py` -> catalog-driven experiment runner
- `experiments/summarize.py` -> aggregate traces and summary figures

Generic training-log analysis lives in `actdyn/utils/training_log_analysis.py` and backs `actdyn analyze`.

## Notes

- Config and registry keys are strict canonical keys (no legacy aliases).
- Use `scripts/migrate_config_keys.py` to migrate older YAML configs.
