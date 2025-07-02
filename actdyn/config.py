from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class EnvironmentConfig:
    env_type: str = "vectorfield"
    dynamics_type: str = "limit_cycle"
    observation_type: str = "loglinear"
    noise_type: str = "poisson"
    action_type: str = "identity"
    latent_dim: int = 2
    observation_dim: int = 2
    action_dim: int = 2
    noise_scale: float = 0.0
    dt: float = 0.1
    device: str = "cuda"


@dataclass
class ModelConfig:
    model_type: str = "seq-vae"
    encoder_type: str = "mlp-encoder"
    dynamics_type: str = "mlp-dynamics"
    observation_type: str = "loglinear"
    noise_type: str = "gaussian"
    observation_dim: int = 2
    action_dim: int = 2
    ensemble: bool = False
    num_models: int = 1
    latent_dim: int = 2
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [16])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [16])
    dynamics_hidden_dims: List[int] = field(default_factory=lambda: [16])


@dataclass
class PolicyConfig:
    policy_type: str = "mpc-icem"
    action_dim: int = 2
    horizon: int = 10
    num_iterations: int = 10
    num_samples: int = 32
    num_elite: int = 100
    alpha: float = 0.1
    device: str = "cuda"


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
    video_path: str = "results/videos"
    model_path: str = "results/models"
    buffer_path: str = "results/buffers"
    logger_path: str = "results/logs"


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda"
    results_dir: str = "results"
    action_dim: int = 2
    observation_dim: int = 2
    latent_dim: int = 2
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        self.model.action_dim = self.action_dim
        self.policy.action_dim = self.action_dim
        self.model.observation_dim = self.observation_dim
        self.model.latent_dim = self.latent_dim

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
        if "training" in config_dict:
            config_dict["training"] = TrainingConfig(**config_dict["training"])
        if "logging" in config_dict:
            config_dict["logging"] = LoggingConfig(**config_dict["logging"])

        return cls(**config_dict)
