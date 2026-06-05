# Active Dynamics - Project Documentation

## I. Project Overview

### Problem Statement
Active Dynamics is a framework for **active learning of latent dynamical systems** through the integration of Sequential Variational Autoencoders (SeqVAEs) with Model Predictive Control (MPC). The system enables efficient identification and control of complex dynamical systems by actively selecting informative data collection trajectories.

### Domain
- **Primary**: Machine Learning / System Identification / Control Theory
- **Sub-domains**: Variational Inference, State-Space Models, Reinforcement Learning, Bayesian Filtering

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT ORCHESTRATION                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  ExperimentConfig (YAML) → setup_experiment() → Experiment.run()        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                  AGENT                                       │
│  Coordinates: Real Environment ↔ Model Environment ↔ Policy                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │ EnvWrapper   │  │ ModelWrapper │  │ Policy       │  │ RecentRollout    │ │
│  │ (Real Env)   │  │ (Latent Sim) │  │ (MPC/Random) │  │ (GPU Buffer)     │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                     │                      │
                     ▼                      ▼
┌────────────────────────────────────┐  ┌─────────────────────────────────────┐
│         ENVIRONMENT LAYER          │  │            MODEL LAYER               │
│  ┌────────────┐  ┌──────────────┐  │  │  ┌─────────────────────────────────┐│
│  │VectorField │  │ EnvWrapper   │  │  │  │           SeqVAE                ││
│  │CartPole    │  │ + ObsModel   │  │  │  │  ┌─────────┐ ┌──────────────┐  ││
│  │Maze        │  │ + ActionModel│  │  │  │  │ Encoder │ │  Dynamics    │  ││
│  └────────────┘  └──────────────┘  │  │  │  │ (RNN)   │ │ (RBF/MLP)    │  ││
│                                    │  │  │  └─────────┘ └──────────────┘  ││
│  Produces: observations y_t        │  │  │  ┌─────────┐ ┌──────────────┐  ││
│  Accepts: actions a_t              │  │  │  │ Decoder │ │ActionEncoder │  ││
│                                    │  │  │  │(LogLin) │ │ (Identity)   │  ││
│                                    │  │  │  └─────────┘ └──────────────┘  ││
└────────────────────────────────────┘  │  └─────────────────────────────────┘│
                                        └─────────────────────────────────────┘
                                                        │
                                                        ▼
                                        ┌─────────────────────────────────────┐
                                        │          POLICY / METRICS           │
                                        │  ┌──────────────┐ ┌───────────────┐ │
                                        │  │  MPC-iCEM    │ │FisherInformation│
                                        │  │  (Planning)  │ │ (A/D-Optimality)│
                                        │  └──────────────┘ └───────────────┘ │
                                        └─────────────────────────────────────┘
```

### Key Workflows

1. **Configuration Loading**
   ```
   config.yaml → ExperimentConfig.from_yaml() → setup_experiment() 
   → (Experiment, Agent, Env, ModelEnv)
   ```

2. **Online Active Learning Loop**
   ```
   For each timestep:
     1. Agent.plan() → Policy selects action using MPC over model
     2. Agent.step() → Execute action in real env, update model state
     3. Agent.train_model() → Update SeqVAE with recent rollout buffer
     4. Metrics updated for next planning iteration
   ```

3. **Offline Learning**
   ```
   Load rollouts from disk → Train SeqVAE via ELBO optimization 
   → Save model checkpoints
   ```

### Data Flow

```
Real Environment                     Model (SeqVAE)
     │                                    │
     │ y_t (observation)                  │ z_t (latent state)
     │ ────────────────────►  Encoder ────►
     │                           │
     │ a_t (action)              │ z_{t+1} (predicted)
     │ ────────────────────►  Dynamics ───►
     │                           │
     │ y_{t+1} (next obs)        │
     │ ◄────────────────────  Decoder ◄───
     │                           │
     └──────── ELBO Training ────┘
```

### Dependencies

**Internal:**
- `actdyn.core`: Agent, Experiment orchestration
- `actdyn.models`: SeqVAE, encoders, decoders, dynamics
- `actdyn.environment`: Gym-compatible environments and wrappers
- `actdyn.policy`: MPC-iCEM and baseline policies  
- `actdyn.metrics`: Fisher Information, costs, rewards
- `actdyn.utils`: Rollout buffers, tensor utilities, plotting

**External:**
- **PyTorch** (≥2.0): Core neural network framework
- **Gymnasium**: Environment interface standard
- **Hydra**: Configuration management (optional)
- **TensorBoard**: Logging and visualization
- **einops**: Tensor manipulation
- **tensordict**: Efficient batched tensor storage
- **colorednoise**: Correlated noise for MPC sampling
- **gpytorch**: Gaussian Process utilities (vector field generation)
- **imageio**: Video recording

---

## II. Module Descriptions & Components

### 1. `actdyn/config.py` - Configuration System

**Purpose:** Dataclass-based configuration for all experiment components. Supports YAML loading and Hydra integration.

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `EnvironmentConfig` | Environment type, dynamics, observation/action models |
| `ModelConfig` | Encoder, decoder, dynamics architecture settings |
| `PolicyConfig` | MPC hyperparameters (horizon, samples, elite selection) |
| `MetricConfig` | Metric types (Fisher, goal, reward) and aggregation |
| `TrainingConfig` | Learning rates, epochs, ELBO β scheduling |
| `LoggingConfig` | Plot/save intervals, video settings |
| `ExperimentConfig` | Top-level config aggregating all sub-configs |

**Key Methods:**
```python
ExperimentConfig.from_yaml(path: str) -> ExperimentConfig
ExperimentConfig.clone() -> ExperimentConfig
ExperimentConfig.to_yaml(path: str) -> None
```

**Design Note:** Uses nested dataclasses with `field(default_factory=...)` for mutable defaults.

---

### 2. `actdyn/core/` - Core Orchestration

#### `agent.py` - Agent Class

**Purpose:** Coordinates interaction between real environment, model environment, and policy.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `Agent` | Standard synchronous agent for Gaussian observation models |
| `AsyncAgent` | Extended agent for Poisson observations with separate prediction/update |

**Key Methods:**
```python
Agent.reset(seed: int) -> torch.Tensor           # Reset env and model
Agent.step(action: torch.Tensor) -> (Transition, bool)  # Execute action
Agent.plan() -> torch.Tensor                     # Get action from policy
Agent.train_model(**kwargs) -> dict              # Train model on recent buffer
Agent.update_policy(rollout: RecentRollout)      # Update policy state
```

**State Tracking:**
- `_observation`: Current observation from environment
- `_env_state`: True latent state (for logging/eval)
- `_model_state`: Model's belief about latent state
- `recent`: GPU-resident rollout buffer for training

#### `experiment.py` - Experiment Orchestration

**Purpose:** Main training loop with logging, checkpointing, and video recording.

**Key Classes:**

| Class | Description |
|-------|-------------|
| `Experiment` | Standard online/offline training experiment |
| `MetaEmbeddingExperiment` | Extended experiment tracking embedding parameters |

**Key Methods:**
```python
Experiment.run(plot_fcn=None, reset=True)       # Online learning loop
Experiment.offline_run(reset=True)              # Offline training from rollouts
Experiment.generate_rollout(num_episodes, ...)  # Create validation data
Experiment.init_experiment(reset=True)          # Setup directories, writer
```

**Features:**
- TensorBoard logging via `SummaryWriter`
- Checkpoint resume from previous runs
- Periodic rollout saving for crash recovery
- Video recording via `VideoRecorder`
- Memory management with explicit CUDA cache clearing

---

### 3. `actdyn/models/` - Model Components

#### `base.py` - Base Classes

**Key Base Classes:**

| Class | Purpose |
|-------|---------|
| `BaseEncoder` | Interface for observation → latent encoding |
| `BaseMapping` | Interface for latent → observation mapping |
| `BaseNoise` | Interface for observation noise models |
| `BaseDynamics` | Interface for latent state dynamics |
| `BaseDynamicsEnsemble` | Wrapper for dynamics model ensembles |
| `BaseModel` | Full model combining encoder, decoder, dynamics |

**BaseDynamics Key Methods:**
```python
compute_param(state) -> (mu, var)           # Mean and variance of dynamics
sample_forward(init_z, action, k_step, return_traj, add_noise) 
    -> (samples, mus, vars)                 # Multi-step rollout
forward(state) -> mu                        # Single-step mean prediction
```

#### `encoder.py` - Encoder Implementations

| Class | Architecture |
|-------|--------------|
| `MLPEncoder` | MLP: obs → hidden → (μ, σ²) |
| `RNNEncoder` | GRU/LSTM over time → (μ, σ²) per timestep |
| `RNNStateEncoder` | State-conditioned RNN encoder |

**Signature:**
```python
forward(y: Tensor, u: Tensor, n_samples: int) -> (samples, mu, var)
```

#### `decoder.py` - Decoder Components

**Mappings (latent → observation mean):**

| Class | Mapping |
|-------|---------|
| `IdentityMapping` | `y = z` |
| `LinearMapping` | `y = Wz + b` |
| `LogLinearMapping` | `y = dt * exp(Wz + b)` (Poisson rates) |
| `MLPMapping` | `y = MLP(z)` |

**Noise Models:**

| Class | Distribution |
|-------|--------------|
| `GaussianNoise` | `y ~ N(μ, σ²)` with learnable σ |
| `PoissonNoise` | `y ~ Poisson(λ)` |

**Decoder Class:**
```python
Decoder(mapping: BaseMapping, noise: BaseNoise, device: str)
    .compute_log_prob(z, x) -> Tensor  # Log-likelihood
    .forward(z) -> Tensor              # Predicted mean
    .jacobian -> Callable              # ∂h/∂z for FIM
```

#### `dynamics.py` - Dynamics Models

| Class | Architecture |
|-------|--------------|
| `LinearDynamics` | `z_{t+1} = Az_t` |
| `FunctionDynamics` | User-defined callable |
| `MLPDynamics` | `z_{t+1} = MLP(z_t)` |
| `RBFDynamics` | `z_{t+1} = z_t + RBF(z_t) * dt` |

**Ensemble Support:**
- `RBFDynamicsEnsemble`
- `MLPDynamicsEnsemble`

**Key Feature:** `is_residual` flag controls whether dynamics predict velocity (`ż`) or next state (`z_{t+1}`).

#### `model.py` - Full Model Classes

| Class | Description |
|-------|-------------|
| `SeqVae` | Sequential VAE with multi-step KL training |
| `SeqStateVae` | State-conditioned variant |
| `DeepVariationalBayesFilter` | DVBF architecture |
| `FilteringEmbedding` | EKF-based filtering with learnable embedding |

**SeqVae Training:**
```python
compute_elbo(y, u, n_samples, k_steps, beta, p_mask, idx)
    -> (-ELBO, log_likelihood, KL)

train_model(data, batch_size, n_epochs, lr, ...)
    -> dict[str, Tensor]  # Training metrics
```

**Multi-step KL:** Computes `KL[q(z_{t+k}) || p_k(z_{t+k}|z_t, u)]` with exponential decay weighting.

#### `model_wrapper.py` - Gym Interface for Models

**Purpose:** Wraps SeqVAE as a gym-compatible environment for MPC planning.

```python
ModelWrapper(model, observation_space, action_space, device)
    .reset(observation) -> (obs, info)
    .step(action) -> (next_obs, reward, terminated, truncated, info)
    .render(ax=None)
```

---

### 4. `actdyn/environment/` - Environment Layer

#### `base.py` - Base Classes

| Class | Purpose |
|-------|---------|
| `BaseAction` | Action encoder interface with validation |
| `BaseObservation` | Observation model with noise injection |

#### `vectorfield.py` - Vector Field Environment

**Purpose:** 2D dynamical systems with various attractors.

**Dynamics Types:**
- `limit_cycle`: Single limit cycle
- `double_limit_cycle`: Nested limit cycles
- `multi_attractor`: GP-generated multi-attractor landscape
- `van_der_pol`: Van der Pol oscillator
- `duffing`: Duffing oscillator
- `snowman`: Custom attractor shape

```python
VectorFieldEnv(dynamics_type, d_state, Q, dt, device, dyn_params, ...)
    .step(action) -> (state, reward, terminated, truncated, info)
    .compute_dynamics(state) -> velocity
```

#### `env_wrapper.py` - Environment Wrapper

**Purpose:** Adds observation/action encoding and converts numpy ↔ torch.

```python
EnvWrapper(env, obs_model, action_model, dt, device)
    .reset() -> (observation, info)   # Returns encoded observation
    .step(action) -> (obs, reward, terminated, truncated, info)
```

**Device Handling:** Auto-detects torch-native environments vs numpy environments.

#### `observation.py` - Observation Models

| Class | Mapping |
|-------|---------|
| `IdentityObservation` | `y = z` |
| `LinearObservation` | `y = Cz + d` |
| `LogLinearObservation` | `y = dt * exp(Cz + d)` |
| `NonlinearObservation` | `y = MLP(z)` |

#### `action.py` - Action Encoders

| Class | Mapping |
|-------|---------|
| `IdentityActionEncoder` | `u' = u` |
| `LinearActionEncoder` | `u' = Wu` |
| `MlpActionEncoder` | `u' = MLP(u)` or `MLP([u, z])` |

---

### 5. `actdyn/policy/` - Policy Layer

#### `base.py` - Base Classes

| Class | Purpose |
|-------|---------|
| `BasePolicy` | Interface with chunking support |
| `BaseMPC` | MPC base with model, metric, horizon |

**Chunking:** Policies can return multi-step action sequences, executing one per call.

#### `mpc.py` - MPC-iCEM Implementation

**Purpose:** Improved Cross-Entropy Method for MPC optimization.

```python
MpcICem(
    num_samples=32,      # Trajectories per iteration
    num_iterations=10,   # CEM refinement iterations
    num_elite=10,        # Elite samples for mean/std update
    alpha=0.1,           # Elite-to-mean interpolation
    init_std=0.5,        # Initial action std
    noise_beta=1.0,      # Colored noise exponent
    horizon=10,          # Planning horizon
    ...
)
```

**Algorithm:**
1. Sample action sequences with colored noise
2. Simulate trajectories through model
3. Evaluate cost via metric
4. Update distribution from elite samples
5. Repeat, decreasing sample count

**Features:**
- Elite action shifting across timesteps
- Elite reuse between iterations
- Adaptive sample size decay

#### `policy.py` - Simple Policies

| Class | Behavior |
|-------|----------|
| `RandomPolicy` | Sample from action space |
| `OffPolicy` | Return pre-specified action sequence |
| `StepPolicy` | Step through action array |

---

### 6. `actdyn/metrics/` - Metrics Layer

#### `base.py` - Base Classes

| Class | Purpose |
|-------|---------|
| `BaseMetric` | Interface with stepwise/aggregate pattern |
| `DiscountedMetric` | Applies temporal discounting |
| `CompositeMetric` | Weighted combination of metrics |

**Pattern:**
```python
metric.compute_stepwise(rollout) -> per_step_costs  # (B, T)
metric.aggregate() -> total_cost                     # (B,)
metric(rollout) -> total_cost                        # Combined call
```

#### `information.py` - Fisher Information Metrics

| Class | Optimality |
|-------|------------|
| `FisherInformationMetric` | Base FIM computation |
| `AOptimality` | `-tr(FIM)` (variance minimization) |
| `DOptimality` | `-log|FIM|` (volume minimization) |
| `EmbeddingFisherMetric` | FIM for embedding parameters |

**FIM Computation:**
```
FIM = ∑_t (∂h/∂z)ᵀ R⁻¹ (∂h/∂z) (∂z/∂θ)ᵀ (∂z/∂θ)
```

**Supports:**
- RBF dynamics parameter sensitivity
- Invariant vs non-invariant covariance
- Diagonal approximations

#### `cost.py` / `reward.py` - Simple Metrics

| Class | Formula |
|-------|---------|
| `ActionCost` | `||a||₂` |
| `RewardMetric` | `-reward` |
| `GoalDistanceMetric` | `||z - z_goal||₂` |

#### `uncertainty.py` - Ensemble Metrics

| Class | Formula |
|-------|---------|
| `EnsembleDisagreement` | Variance across ensemble predictions |

---

### 7. `actdyn/utils/` - Utilities

#### `rollout.py` - Data Storage

| Class | Purpose |
|-------|---------|
| `Rollout` | Single trajectory storage |
| `RolloutBuffer` | Collection of rollouts with DataLoader support |
| `RecentRollout` | Fixed-size GPU buffer for online training |

**Rollout Fields:** `obs`, `next_obs`, `action`, `env_action`, `reward`, `env_state`, `next_env_state`, `model_state`, `next_model_state`, `model_action`

**Key Methods:**
```python
Rollout.add(**kwargs)           # Add transition
Rollout.finalize()              # Convert to tensors
Rollout.downsample(n)           # Keep every n-th step
Rollout.copy()                  # Deep copy

RolloutBuffer.get_dataloader(batch_size, chunk_size, shuffle)
```

#### `experiment_setup.py` - Setup Functions

**Main Entry Point:**
```python
setup_experiment(config: ExperimentConfig) 
    -> (Experiment, Agent, EnvWrapper, ModelWrapper)
```

**Component Setup:**
```python
setup_environment(config) -> EnvWrapper
setup_model(config) -> SeqVae
setup_metric(config, model) -> BaseMetric
setup_policy(config, env, model, metric) -> BasePolicy
```

#### `torch_utils.py` - General Utilities

| Function | Purpose |
|----------|---------|
| `to_np(tensor)` | Convert to numpy |
| `safe_cholesky(M, jitter)` | Robust Cholesky decomposition |
| `symmetrize(M)` | Ensure matrix symmetry |
| `activation_from_str(name)` | String → activation function |
| `format_list(x)` | Pretty-print for logging |

#### `vectorfields_eqn.py` - Dynamics Definitions

| Class | Dynamics |
|-------|----------|
| `LimitCycle` | `ż = α(d - r²)z + ω⊥z` |
| `DoubleLimitCycle` | Nested ring attractors |
| `MultiAttractor` | GP-sampled potential field |
| `VanDerPol` | `ẍ = μ(1-x²)ẋ - x` |
| `Duffing` | `ẍ = -δẋ - αx - βx³` |

#### `plotting.py` - Plotting Utilities

```python
plot_vector_field(dynamics, x_range, n_grid, device, ax)
compute_vector_field(dynamics, ...) -> (X, Y, U, V)
create_grid(x_range, n_grid, device) -> (grid, xx, yy)
set_matplotlib_style()  # Publication-ready defaults
```

#### `video.py` - Video Recording

```python
VideoRecorder(path, fps, codec)
    .capture_frame(fig)  # Add matplotlib figure
    .close()             # Write video file
```

**Codecs:** `h264` (default), `prores` (Keynote), `lossless`

#### `persistence.py` - Persistence

```python
save_rollout(rollout, path)
load_rollout(path) -> Rollout
save_config(config, path)
load_and_concatenate_rollouts(dir, pattern) -> Rollout
```

#### `hydra_config.py` - Hydra Support

```python
@hydra_experiment(config_path="conf")
def main(config: ExperimentConfig):
    ...

HydraExperimentConfig.from_hydra_dict(cfg) -> ExperimentConfig
register_actdyn_configs()  # Register with ConfigStore
```

---

### 8. `experiments/` - Experiment Scripts

#### Structure
```
experiments/
├── experiment_definitions.py # Environment, policy, schedule, and suite definitions
├── experiment_io.py      # Metadata, artifact, and argument parsing helpers
├── run.py                 # Catalog-driven experiment runner
├── summarize.py           # Trace aggregation and summary figures
├── render_videos.py       # Experiment video rendering
├── tbme/                  # TBME experiment suites and figures
├── _hydra_templates/      # Shared config templates
```

#### Running Experiments
```bash
# Primary entry point
uv run actdyn run --config experiments/active_embedding/conf/config.yaml

# Sweep
uv run actdyn sweep --config-path experiments/ciss/conf

# Analyze
uv run actdyn analyze results

```

---

### 9. `external/` - External Dependencies

Contains `integrative_inference` submodule for additional filtering algorithms.

---

### 10. `tests/` - Test Suite

Current baseline focuses on smoke + contract coverage for the refactored entry points and registry behavior.

| File | Coverage |
|------|----------|
| `tests/test_cli_smoke.py` | CLI parse/help + tiny run smoke |
| `tests/test_environment_registry.py` | Canonical environment key contract |

**Running Tests:**
```bash
uv run pytest -q
uv run python -m compileall -q actdyn experiments tests
```

---

## III. API Guide

### Public-Facing APIs

#### 1. Experiment Setup (Primary Entry Point)

```python
from actdyn.utils import setup_experiment
from actdyn.config import ExperimentConfig

# From YAML
config = ExperimentConfig.from_yaml("path/to/config.yaml")

# Setup all components
experiment, agent, env, model_env = setup_experiment(config)

# Run online training
experiment.run()

# Run offline training
experiment.offline_run()
```

#### 2. Direct Component Usage

```python
from actdyn.environment import environment_from_str, EnvWrapper
from actdyn.models import SeqVae, encoder_from_str, dynamics_from_str
from actdyn.policy import policy_from_str
from actdyn.metrics import metric_from_str

# Create environment
env_cls = environment_from_str("vectorfield")
base_env = env_cls(dynamics_type="limit_cycle", dt=0.1)

# Create model
encoder = encoder_from_str("rnn")(obs_dim=50, latent_dim=2, ...)
decoder = Decoder(mapping, noise, device="cuda")
dynamics = dynamics_from_str("rbf")(state_dim=2, ...)
model = SeqVae(encoder=encoder, decoder=decoder, dynamics=dynamics)

# Create policy  
metric = metric_from_str("a-optimality")(model=model, ...)
policy = policy_from_str("mpc-icem")(model=model, metric=metric, horizon=10)
```

### Configuration Mechanisms

#### YAML Configuration
```yaml
seed: 42
device: cuda
latent_dim: 2
observation_dim: 50
dt: 0.1

environment:
  environment_type: vectorfield
  env_dynamics_type: limit_cycle
  observation_type: log-linear
  obs_noise_type: poisson

model:
  encoder_type: rnn
  dynamics_type: rbf
  mapping_type: log-linear
  noise_type: poisson

policy:
  policy_type: mpc-icem
  mpc_horizon: 10
  mpc_num_samples: 32

metric:
  metric_type: [a-optimality, action]
  compute_type: sum

training:
  total_steps: 10000
  learning_rate: 1e-3
  n_epochs: 1
  rollout_horizon: 20
```

#### CLI Invocation
```bash
uv run actdyn run \
  --config experiments/active_embedding/conf/config.yaml \
  --seed 1 \
  --device cpu

uv run actdyn sweep \
  --config-path experiments/ciss/conf \
  --dry-run
```

### Command-Line Entry Points

| Script | Purpose |
|--------|---------|
| `uv run actdyn` | Canonical CLI (`run`, `sweep`, `analyze`) |
| `experiments/run.py` | Catalog-driven experiment runner |
| `experiments/summarize.py` | Summary CSV and figure generation |
| `experiments/render_videos.py` | Experiment video rendering |
| `actdyn/utils/training_log_analysis.py` | Generic training-log analysis backend for `actdyn analyze` |

### Expected Inputs/Outputs

#### Agent.step()
```python
# Input
action: torch.Tensor  # Shape: (1, 1, action_dim)

# Output
transition: dict = {
    'obs': Tensor,           # (1, 1, obs_dim)
    'next_obs': Tensor,      # (1, 1, obs_dim)
    'action': Tensor,        # (1, 1, action_dim)
    'env_action': Tensor,    # (1, 1, latent_dim) encoded
    'reward': float,
    'env_state': Tensor,     # (1, 1, latent_dim) true state
    'model_state': Tensor,   # (1, 1, latent_dim) belief
    ...
}
done: bool
```

#### SeqVae.compute_elbo()
```python
# Inputs
y: Tensor           # (batch, time, obs_dim)
u: Tensor | None    # (batch, time, action_dim)
n_samples: int      # MC samples for posterior
k_steps: int        # Multi-step KL horizon
beta: float         # KL weight

# Outputs
-elbo: Tensor       # Scalar loss
log_likelihood: Tensor
kl_divergence: Tensor
```

#### Metric.__call__()
```python
# Input
rollout: Rollout | RolloutBuffer | dict
    # Must contain: 'model_state', 'next_model_state', 'action'

# Output
cost: Tensor  # Shape: (batch,)
```

### Error Handling

- **Device Mismatch:** Auto-fallback from CUDA to CPU if unavailable
- **Config Validation:** Dataclass field validation on construction
- **Tensor Shape Errors:** Explicit assertions with informative messages
- **Gradient Overflow:** Configurable gradient clipping (`grad_clip_norm`)
- **Memory Management:** Explicit `torch.cuda.empty_cache()` in training loops

### Known Limitations / Experimental Features

1. **`FilteringEmbedding`:** Experimental EKF-based model, API may change
2. **Maze environments:** Partially implemented (`maze_environment_*.py`)
3. **Async filtering:** `AsyncAgent` designed for Poisson observations only
4. **Model save/load:** Not fully implemented in `SeqVae` (see DESIGN NOTEs)
5. **Multi-GPU:** Not currently supported

---

## IV. File Index

### Core Package (`actdyn/`)
| File | Lines | Description |
|------|-------|-------------|
| `config.py` | 344 | Dataclass configurations |
| `core/agent.py` | ~180 | Agent orchestration |
| `core/experiment.py` | 465 | Training loops |
| `models/base.py` | 629 | Base model classes |
| `models/model.py` | 1114 | SeqVAE implementations |
| `models/encoder.py` | 642 | Encoder architectures |
| `models/decoder.py` | 230 | Decoder components |
| `models/dynamics.py` | 300 | Dynamics models |
| `models/model_wrapper.py` | 130 | Gym wrapper for models |
| `environment/vectorfield.py` | 145 | Vector field env |
| `environment/env_wrapper.py` | 130 | Environment wrapper |
| `policy/mpc.py` | 250 | MPC-iCEM |
| `policy/base.py` | 95 | Policy base classes |
| `metrics/information.py` | 551 | Fisher Information |
| `metrics/base.py` | 100 | Metric base classes |
| `utils/rollout.py` | 909 | Data storage |
| `utils/experiment_setup.py` | 286 | Setup functions |
| `utils/plotting.py` | 270 | Plotting |
| `utils/hydra_config.py` | 160 | Hydra support |

### Tests (`tests/`)
| File | Tests | Coverage |
|------|-------|----------|
| `test_config.py` | Config parsing, YAML loading |
| `test_models_encoder.py` | MLP/RNN encoders |
| `test_models_decoder.py` | Mappings, noise models |
| `test_models_dynamics.py` | All dynamics variants |
| `test_models_model.py` | SeqVAE training |
| `test_environment.py` | Environments, wrappers |
| `test_core_agent.py` | Agent step/reset |
| `test_core_experiment.py` | Experiment orchestration |
| `test_policy.py` | Policy implementations |
| `test_mpc.py` | MPC-iCEM specifics |
| `test_utils_rollout.py` | Rollout buffers |

---

## V. Quick Reference

### Factory Functions

| Function | Returns | Keys |
|----------|---------|------|
| `environment_from_str(key)` | `type[Env]` | `vectorfield`, `windfield` |
| `observation_from_str(key)` | `type[BaseObservation]` | `identity`, `linear`, `log-linear`, `non-linear` |
| `action_from_str(key)` | `type[BaseAction]` | `identity`, `linear`, `mlp` |
| `encoder_from_str(key)` | `type[BaseEncoder]` | `mlp`, `rnn` |
| `dynamics_from_str(key)` | `type[BaseDynamics]` | `linear`, `mlp`, `rbf` |
| `mapping_from_str(key)` | `type[BaseMapping]` | `identity`, `linear`, `log-linear`, `mlp` |
| `noise_from_str(key)` | `type[BaseNoise]` | `gaussian`, `poisson` |
| `model_from_str(key)` | `type[BaseModel]` | `seq-vae`, `filtering-embedding` |
| `policy_from_str(key)` | `type[BasePolicy]` | `random`, `off-policy`, `mpc-icem` |
| `metric_from_str(key)` | `type[BaseMetric]` | `a-optimality`, `d-optimality`, `action`, `goal-distance`, `reward` |

### uv Environment
```bash
uv sync
uv run actdyn --help
```

### Common Imports
```python
from actdyn import (
    Agent, Experiment, ExperimentConfig, setup_experiment,
    SeqVae, ModelWrapper, VectorFieldEnv, EnvWrapper,
    MpcICem, FisherInformationMetric
)
```

---

## VI. TODO / Roadmap

- [ ] Add a minimal, CPU-only quickstart config and tutorial.
- [ ] Document the action/observation naming convention and mapping between `d_*` and `*_dim`.
- [ ] Summarize known test isolation issues and recommended pytest flags.
- [ ] Provide a table of supported environment/model combinations.
