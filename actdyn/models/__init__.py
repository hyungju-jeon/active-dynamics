from .base import (
    BaseModel,
    BaseEncoder,
    BaseDynamics,
    BaseMapping,
    BaseNoise,
    EnsembleDynamics,
)
from .decoder import Decoder
from .model_wrapper import VAEWrapper
import importlib

__all__ = [
    "mapping_from_str",
    "noise_from_str",
    "encoder_from_str",
    "dynamics_from_str",
    "model_from_str",
    "Decoder",
    "VAEWrapper",
    "EnsembleDynamics",
]

_model_map = {
    "seq-vae": (".model", "SeqVae"),
    # Add more mappings as needed
}

_encoder_map = {
    "mlp": (".encoder", "MLPEncoder"),
    "rnn": (".encoder", "RNNEncoder"),
    # Add more mappings as needed
}

_mapping_map = {
    "identity": (".decoder", "IdentityMapping"),
    "linear": (".decoder", "LinearMapping"),
    "loglinear": (".decoder", "LogLinearMapping"),
    "mlp": (".decoder", "MLPMapping"),
    # Add more mappings as needed
}

_noise_map = {
    "gaussian": (".decoder", "GaussianNoise"),
    "poisson": (".decoder", "PoissonNoise"),
    # Add more mappings as needed
}

_dynamics_map = {
    "linear": (".dynamics", "LinearDynamics"),
    "mlp": (".dynamics", "MLPDynamics"),
    "rbf": (".dynamics", "RBFDynamics"),
    # Add more mappings as needed
}


def mapping_from_str(mapping_str: str) -> type[BaseMapping]:
    """
    Dynamically import and return the mapping class based on the string key.
    Example: mapping_from_string('mlp')
    """
    if mapping_str not in _mapping_map:
        raise ImportError(
            f"Unknown mapping: {mapping_str}. Available: {list(_mapping_map.keys())}"
        )
    module_name, class_name = _mapping_map[mapping_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def noise_from_str(noise_str: str) -> type[BaseNoise]:
    """
    Dynamically import and return the noise class based on the string key.
    Example: noise_from_string('gaussian')
    """
    if noise_str not in _noise_map:
        raise ImportError(
            f"Unknown noise: {noise_str}. Available: {list(_noise_map.keys())}"
        )
    module_name, class_name = _noise_map[noise_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def encoder_from_str(encoder_str: str) -> type[BaseEncoder]:
    """
    Dynamically import and return the encoder class based on the string key.
    Example: encoder_from_string('mlp-encoder')
    """
    if encoder_str not in _encoder_map:
        raise ImportError(
            f"Unknown encoder: {encoder_str}. Available: {list(_encoder_map.keys())}"
        )
    module_name, class_name = _encoder_map[encoder_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def dynamics_from_str(dynamics_str: str) -> type[BaseDynamics]:
    """
    Dynamically import and return the dynamics class based on the string key.
    Example: dynamics_from_string('linear-dynamics')
    """
    if dynamics_str not in _dynamics_map:
        raise ImportError(
            f"Unknown dynamics: {dynamics_str}. Available: {list(_dynamics_map.keys())}"
        )
    module_name, class_name = _dynamics_map[dynamics_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)


def model_from_str(model_str: str) -> type[BaseModel]:
    """
    Dynamically import and return the model class based on the string key.
    Example: model_from_string('seq-vae')
    """
    if model_str not in _model_map:
        raise ImportError(
            f"Unknown model: {model_str}. Available: {list(_model_map.keys())}"
        )
    module_name, class_name = _model_map[model_str]
    module = importlib.import_module(module_name, __package__)
    return getattr(module, class_name)
