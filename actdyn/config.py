from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import torch
import yaml


@dataclass
class EnvironmentConfig:
    environment_type: str = "vectorfield"
    env_dynamics_type: Optional[str] = "limit_cycle"
    env_dt: float = 0.1
    env_noise_scale: float = 0.1
    env_render_mode: Optional[str] = None
    env_action_bounds: Sequence[float] = field(default_factory=lambda: [-0.1, 0.1])
    env_state_bounds: Optional[Sequence[float]] = None

    observation_type: str = "loglinear"
    obs_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [16])
    obs_activation: Optional[str] = "relu"
    noise_type: Optional[str] = "gaussian"
    noise_scale: float = 0.0

    action_type: str = "identity"
    act_action_bounds: Sequence[float] = field(default_factory=lambda: [-0.1, 0.1])
    act_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [16])
    act_activation: Optional[str] = "relu"


@dataclass
class ModelConfig:
    encoder_type: str = "rnn"
    enc_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [16])
    enc_hidden_dim: Optional[int] = 16
    enc_rnn_type: Optional[str] = "gru"
    enc_num_layers: Optional[int] = 1
    enc_activation: Optional[str] = "relu"

    mapping_type: str = "loglinear"
    map_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [16])
    map_activation: Optional[str] = "relu"
    noise_type: str = "gaussian"

    dynamics_type: str = "mlp"
    is_ensemble: bool = False
    n_models: Optional[int] = 1
    dyn_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [16])
    dyn_alpha: float = 0.1
    dyn_gamma: float = 1.0
    dyn_num_centers: Optional[int] = None

    action_type: str = "identity"
    act_action_bounds: Sequence[float] = field(default_factory=lambda: [-0.1, 0.1])
    act_hidden_dims: Optional[List[int]] = field(default_factory=lambda: [16])
    act_activation: Optional[str] = "relu"

    model_type: str = "seq-vae"


@dataclass
class PolicyConfig:
    policy_type: str = "random"
    mpc_verbose: bool = False
    mpc_horizon: int = 10
    mpc_num_samples: int = 32
    mpc_num_iterations: int = 10
    mpc_num_elite: int = 100
    mpc_alpha: float = 0.1
    mpc_init_std: float = 0.5
    mpc_noise_beta: float = 1.0
    mpc_factor_decrease_num: float = 1.25
    mpc_frac_prev_elites: float = 0.2
    mpc_frac_elites_reused: float = 0.3
    mpc_use_mean_actions: bool = True
    mpc_shift_elites: bool = True
    mpc_keep_elites: bool = True


@dataclass
class MetricConfig:
    metric_type: [str, List[str]] = "A-optimality"
    compute_type: str = "sum"
    gamma: [float, List[float]] = 1.0
    composite_weights: Optional[List[float]] = None
    met_goal: Optional[List[float]] = None
    met_use_diag: Optional[bool] = False

    def __post_init__(self):
        self.met_goal = (
            torch.tensor(self.met_goal) if self.met_goal is not None else None
        )


@dataclass
class TrainingConfig:
    total_steps: int = 10000
    train_every: int = 1
    save_every: int = 1000
    animate_every: int = 1000
    rollout_horizon: int = 20
    batch_size: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class LoggingConfig:
    log_every: int = 100
    save_every: int = 1000
    video_path: Optional[str] = None
    model_path: str = "results/models"
    buffer_path: str = "results/buffers"
    logger_path: str = "results/logs"


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cpu"
    results_dir: str = "results"
    action_dim: int = 2
    observation_dim: int = 50
    latent_dim: int = 2
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Create an ExperimentConfig instance from a YAML file."""

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert nested dictionaries to their respective config classes
        if "environment" in config_dict:
            config_dict["environment"] = EnvironmentConfig(**config_dict["environment"])
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "policy" in config_dict:
            config_dict["policy"] = PolicyConfig(**config_dict["policy"])
        if "metric" in config_dict:
            config_dict["metric"] = MetricConfig(**config_dict["metric"])
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        if "logging" in config_dict:
            config_dict["logging"] = LoggingConfig(**config_dict["logging"])

        return cls(**config_dict)
