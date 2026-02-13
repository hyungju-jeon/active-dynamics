# Active Dynamics

Active learning framework for latent dynamical system identification.

## Installation

```bash
conda env create -f environment.yml
conda activate active-dynamics
pip install -e .
```

## Supported Experiment Tracks

- `experiments/active_embedding`
- `experiments/ciss`
- `experiments/_hydra_templates`

## Single Entry Point

Use the package CLI as the standard runner:

```bash
python -m actdyn --help
python -m actdyn run --config experiments/active_embedding/conf/config.yaml
python -m actdyn sweep --config-path experiments/ciss/conf
python -m actdyn analyze results
python -m actdyn analyze results --summary --save-summary
```

Legacy wrappers still exist and forward to the CLI:

- `experiments/run_experiment.py` -> `actdyn run`
- `experiments/run_hydra.py` -> `actdyn sweep`

## Notes

- Config and registry keys are strict canonical keys (no legacy aliases).
- Use `scripts/migrate_config_keys.py` to migrate older YAML configs.
