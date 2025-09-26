import copy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
import yaml


@dataclass
class EnvironmentConfig:
    environment_type: str = "vectorfield"  # Options: "vectorfield", "cartpole", "maze"
    env_dynamics_type: Optional[str] = (
        "limit_cycle"  # Options: "limit_cycle", "double_limit_cycle", "multi_attractor"
    )
    env_dt = 0.1
    env_noise_scale: float = 0.1
    env_render_mode: Optional[str] = None  # Options: "human", "rgb_array"
    env_action_bounds: List[float] = field(default_factory=lambda: [-0.1, 0.1])
    env_state_bounds: Optional[List[float]] = None
    env_x_range: float = 2.0  # Range for vectorfield environment
    env_n_grid: int = 40
    env_w_attractor: float = 0.1
    env_length_scale: float = 0.5
    env_alpha: float = 0.25  # Scaling factor for the vector field

    observation_type: str = "loglinear"  # Options: "loglinear", "linear", "identity"
    obs_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    obs_activation: Optional[str] = "relu"  # Options: "relu", "tanh", "sigmoid", "leaky_relu"
    obs_noise_type: Optional[str] = "gaussian"  # Options: "gaussian", "poisson"
    obs_noise_scale: float = 0.0

    action_type: str = "identity"  # Options: "identity", "linear", "mlp"
    act_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    act_activation: Optional[str] = "relu"  # Options: "relu", "tanh", "sigmoid", "leaky_relu"

    def get_environment_cfg(self):
        return {
            "dt": self.env_dt,
            "noise_scale": self.env_noise_scale,
            "render_mode": self.env_render_mode,
            "action_bounds": self.env_action_bounds,
            "state_bounds": self.env_state_bounds,
            "x_range": self.env_x_range,
            "n_grid": self.env_n_grid,
            "w_attractor": self.env_w_attractor,
            "length_scale": self.env_length_scale,
            "alpha": self.env_alpha,
        }

    def get_observation_cfg(self):
        return {
            "hidden_dim": self.obs_hidden_dim,
            "activation": self.obs_activation,
            "noise_type": self.obs_noise_type,
            "noise_scale": self.obs_noise_scale,
        }

    def get_action_cfg(self):
        return {
            "hidden_dim": self.act_hidden_dim,
            "activation": self.act_activation,
        }


@dataclass
class ModelConfig:
    encoder_type: str = "rnn"  # Options: "rnn", "mlp"
    enc_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    enc_rnn_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    enc_rnn_type: Optional[str] = "gru"  # Options: "gru", "lstm"
    enc_activation: Optional[str] = "relu"  # Options: "relu", "tanh", "sigmoid", "leaky_relu"
    enc_h_init: Optional[str] = "reset"  # Options: "reset", "carryover", "step"

    mapping_type: str = "loglinear"  # Options: "loglinear", "linear", "identity", "mlp"
    map_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    map_activation: Optional[str] = "relu"  # Options: "relu", "tanh", "sigmoid", "leaky_relu"
    noise_type: str = "gaussian"  # Options: "gaussian", "poisson"

    dynamics_type: str = "mlp"  # Options: "mlp", "linear", "rbf"
    is_ensemble: bool = False
    is_residual: bool = False
    n_models: Optional[int] = 1
    dyn_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    dyn_activation: Optional[str] = "relu"  # Options: "relu", "tanh", "sigmoid", "leaky_relu"
    dyn_alpha: float = 0.1
    dyn_gamma: float = 1.0
    dyn_centers: Optional[List[List[float]]] = None  # Will be converted to tensor when needed
    dyn_range: float = 2.0
    dyn_num_grid: int = 25
    dyn_dt: float = 0.1

    action_type: str = "identity"  # Options: "identity", "linear", "mlp"
    act_hidden_dim: Optional[List[int]] = field(default_factory=lambda: [16])
    act_activation: Optional[str] = "relu"  # Options: "relu", "tanh", "sigmoid", "leaky_relu"
    act_state_dependent: bool = False
    model_type: str = "seq-vae"  # Options: "seq-vae"

    def get_encoder_cfg(self):
        return {
            "hidden_dim": self.enc_hidden_dim,
            "rnn_hidden_dim": self.enc_rnn_hidden_dim,
            "activation": self.enc_activation,
            "rnn_type": self.enc_rnn_type,
            "h_init": self.enc_h_init,
        }

    def get_decoder_cfg(self):
        return {
            "hidden_dim": self.map_hidden_dim,
            "activation": self.map_activation,
        }

    def get_dynamics_cfg(self):
        return {
            "hidden_dim": self.dyn_hidden_dim,
            "activation": self.dyn_activation,
            "alpha": self.dyn_alpha,
            "gamma": self.dyn_gamma,
            "centers": self.dyn_centers,
            "z_max": self.dyn_range,
            "num_grid": self.dyn_num_grid,
            "is_residual": self.is_residual,
            "dt": self.dyn_dt,
        }

    def get_action_cfg(self):
        return {
            "hidden_dim": self.act_hidden_dim,
            "activation": self.act_activation,
            "state_dependent": self.act_state_dependent,
        }

    def get_ensemble_cfg(self):
        return {
            "n_models": self.n_models,
        }


@dataclass
class PolicyConfig:
    policy_type: str = "random"  # Options: "random", "lazy", "mpc-icem"
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

    def get_mpc_cfg(self):
        return {
            "verbose": self.mpc_verbose,
            "horizon": self.mpc_horizon,
            "num_samples": self.mpc_num_samples,
            "num_iterations": self.mpc_num_iterations,
            "num_elite": self.mpc_num_elite,
            "alpha": self.mpc_alpha,
            "init_std": self.mpc_init_std,
            "noise_beta": self.mpc_noise_beta,
            "factor_decrease_num": self.mpc_factor_decrease_num,
            "frac_prev_elites": self.mpc_frac_prev_elites,
            "frac_elites_reused": self.mpc_frac_elites_reused,
            "use_mean_actions": self.mpc_use_mean_actions,
            "shift_elites": self.mpc_shift_elites,
            "keep_elites": self.mpc_keep_elites,
        }


@dataclass
class MetricConfig:
    metric_type: List[str] = field(
        default_factory=lambda: ["A-optimality"]
    )  # Options: "A-optimality", "D-optimality", "action", "goal", "reward"
    compute_type: str = "sum"  # Options: "sum", "last", "max"
    gamma: float = 1.0  # Can be a single value, will be handled as list when needed
    composite_weights: Optional[List[float]] = None
    met_goal: Optional[List[float]] = None  # Will be converted to tensor when needed
    met_discount_factor: float = 0.99
    met_use_diag: Optional[bool] = False
    met_sensitivity: bool = True
    met_covariance: str = "invariant"  # Options: "invariant", "1st", "deterministic"

    def get_metric_cfg(self):
        return {
            "compute_type": self.compute_type,
            "met_goal": self.met_goal,
            "met_discount_factor": self.met_discount_factor,
            "met_use_diag": self.met_use_diag,
            "met_sensitivity": self.met_sensitivity,
            "met_covariance": self.met_covariance,
        }


@dataclass
class TrainingConfig:
    # Training parameters
    total_steps: int = 10000
    train_every: int = 1
    rollout_horizon: int = 20
    p_mask: float = 0.0

    # ELBO optimizier configuration
    beta: float = 1.0
    optimizer: str = "SGD"  # Options: "SGD", "Adam", "AdamW"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    n_epochs: int = 1
    verbose: bool = False
    grad_clip_norm: float = 0.0
    n_samples: int = 5
    k_steps: int = 5
    perturbation: float = 0.01
    annealing_type: str = "none"  # Options: "linear", "cyclic", "none"
    annealing_steps: int = 1000
    warmup: int = 0
    # Offline training parameters
    offline_lr: float = 1e-4
    offline_n_epochs: int = 5000
    offline_batch_size: int = 32
    offline_chunk_size: int = 1000
    offline_decay: float = 1e-4
    offline_annealing_type: str = "none"
    offline_annealing_steps: int = 2000
    offline_warmup: int = 0
    param_list: Any = "all"  # List of parameters to log

    def get_offline_optim_cfg(self):
        return {
            "lr": self.offline_lr,
            "n_epochs": self.offline_n_epochs,
            "batch_size": self.offline_batch_size,
            "chunk_size": self.offline_chunk_size,
            "weight_decay": self.offline_decay,
            "n_samples": self.n_samples,
            "k_steps": self.k_steps,
            "optimizer": "AdamW",
            "perturbation": 0.0,
            "grad_clip_norm": self.grad_clip_norm,
            "annealing_type": self.offline_annealing_type,
            "annealing_steps": self.offline_annealing_steps,
            "warmup": self.offline_warmup,
            "beta": self.beta,
            "shuffle": False,
            "verbose": True,
            "param_list": self.param_list,
            "p_mask": self.p_mask,
        }

    def get_optim_cfg(self):
        return {
            "optimizer": self.optimizer,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "n_epochs": self.n_epochs,
            "perturbation": self.perturbation,
            "grad_clip_norm": self.grad_clip_norm,
            "annealing_type": self.annealing_type,
            "annealing_steps": self.annealing_steps,
            "warmup": self.warmup,
            "n_samples": self.n_samples,
            "k_steps": self.k_steps,
            "verbose": self.verbose,
            "beta": self.beta,
            "param_list": self.param_list,
            "p_mask": self.p_mask,
        }


@dataclass
class LoggingConfig:
    plot_every: int = 1000
    save_every: int = 1000
    video_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "cpu"  # Options: "cpu", "cuda", "mps"
    results_dir: str = "results"
    data_dir: str = "data"
    action_dim: int = 2
    observation_dim: int = 50
    latent_dim: int = 2
    dt: float = 0.1
    run_online: bool = True
    run_offline: bool = True
    run_analysis: bool = False
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Create an ExperimentConfig instance from a YAML file."""

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # Filter out Hydra-specific sections that aren't part of our dataclass
        hydra_keys = {"defaults", "hydra"}
        config_dict = {k: v for k, v in config_dict.items() if k not in hydra_keys}
        # Convert nested dictionaries to their respective config classes
        if "environment" in config_dict:
            config_dict["environment"] = EnvironmentConfig(**config_dict["environment"])
            config_dict["environment"].env_dt = config_dict["dt"]
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

    def clone(self) -> "ExperimentConfig":
        """Create a deep copy of the experiment configuration."""
        return copy.deepcopy(self)

    def to_yaml(self, yaml_path: str) -> None:
        """Save the ExperimentConfig instance to a YAML file."""
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)

    def to_dict(self) -> dict:
        """Convert the ExperimentConfig instance to a dictionary."""
        return self.__dict__
