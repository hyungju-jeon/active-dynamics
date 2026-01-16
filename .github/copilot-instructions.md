# Active Dynamics - AI Coding Assistant Instructions

## Project Overview
Active learning framework for latent dynamical system identification that couples Sequential VAEs with MPC to learn and control complex systems via active data collection.

## Key Architecture Components

### 1. Hierarchical Component Structure
- **Core**: `Agent` coordinates real/model environments and policies.
- **Environment**: `VectorFieldEnv` (real dynamics) wrapped by `EnvWrapper` (observation/action encoding + dt handling).
- **Model**: `SeqVae` (encoder + dynamics + decoder) wrapped by `ModelWrapper` (gym-like interface for latent rollouts).
- **Policy**: MPC-based policies drive exploration and control.
- **Experiment**: Training loop with GPU-aware memory management and progress tracking.

### 2. Critical Configuration Pattern
Always load YAML configs through `ExperimentConfig.from_yaml()` (nested dataclasses) and set up via `setup_experiment()`:
```python
config = ExperimentConfig.from_yaml("path/to/config.yaml")
experiment, agent, env, model_env = setup_experiment(config)
```
Key config sections: `environment`, `model`, `policy`, `metric`, `training`, `logging`. `setup_experiment()` also falls back to CPU when `config.device="cuda"` but CUDA is unavailable.

### 3. Model Training & Memory Management
Use the recent rollout buffer on GPU and guard against CUDA OOM:
```python
agent.train_model(optimizer="SGD", lr=1e-3, n_epochs=1, grad_clip_val=1.0)

if 'cuda' in str(device):
    torch.cuda.empty_cache()
```
Wrap inference in `torch.no_grad()`, explicitly delete tensors in loops, and clip gradients as configured.

### 4. SeqVAE Architecture
`SeqVae` combines:
- **Encoder**: RNN/MLP mapping observations → latent distributions.
- **Dynamics**: Predicts next latent given current state + action (ensemble optionally for disagreement metrics).
- **Decoder**: `mapping_from_str` + `noise_from_str` to map latent states → observations with learned noise.
- **Action Encoder**: Maps raw actions → latent action space.
Training uses ELBO with KL between predicted and actual transitions.

### 5. Environment Wrapping Pattern
Wrap real and model environments:
```python
VectorFieldEnv -> EnvWrapper -> Agent.env
SeqVae -> ModelWrapper -> Agent.model_env
```
The `Agent` steps both to keep real vs. model state estimates aligned.

## Development Patterns

### Experiment Creation
1. Copy an existing config in `experiments/` and adjust parameters.
2. Always use `setup_experiment()`; do not manually wire components.
3. Run with `experiment.run()` for training, logging, and memory cleanup.

### Component Selection
Use factory helpers:
- `environment_from_str("vectorfield")`
- `encoder_from_str("rnn")`, `dynamics_from_str("rbf")`
- `mapping_from_str(...)`, `noise_from_str(...)`, `model_from_str(...)`
- `metric_from_str(...)` (supports lists + `CompositeMetric`)
- `policy_from_str("lazy")` or `policy_from_str("mpc")` (MPC requires a metric and model)

### File Structure Conventions
- Configs: `experiments/[experiment_name]/config_*.yaml`
- Results: `results/[experiment_name]/`
- Core code: `actdyn/[core|models|environment|policy]/`
- Orchestration: `actdyn.utils.helpers.setup_experiment()` only; never manual wiring
Follow `experiments/vectorfield_test/` for end-to-end setup and memory handling.

Always use the `active-dynamics` conda environment:
```bash
conda activate active-dynamics
```
