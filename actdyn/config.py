from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvironmentConfig:
    dynamics_type: str = "limit_cycle"  # Options: "limit_cycle", "double_limit_cycle"
    dim: int = 2
    noise_scale: float = 0.1
    dt: float = 0.1
    device: str = "cuda"


@dataclass
class ModelConfig:
    input_dim: int = 2
    latent_dim: int = 2
    num_models: int = 1
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [16])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [16])
    dynamics_hidden_dims: List[int] = field(default_factory=lambda: [16])


@dataclass
class PolicyConfig:
    device: str = "cuda"
    horizon: int = 10
    num_iterations: int = 10
    num_samples: int = 32
    num_elite: int = 100
    alpha: float = 0.1


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
    action_dim: int = 2  # Single source of truth for action_dim
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self):
        # Ensure model and policy configs use the same action_dim
        self.model.action_dim = self.action_dim
        self.policy.action_dim = self.action_dim

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Create an ExperimentConfig instance from a YAML file."""
        import yaml

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
