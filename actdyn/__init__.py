"""
Active Dynamics: Active Learning for Latent Dynamical System Identification

A framework for active learning of latent dynamical systems using Sequential VAEs
and Model Predictive Control.
"""

__version__ = "0.1.0"
__author__ = "Hyungju Jeon"

# Core components
from actdyn.core.agent import Agent
from actdyn.core.experiment import Experiment

# Configuration
from actdyn.config import ExperimentConfig

# Utilities
from actdyn.utils.experiment_helpers import setup_experiment

# Main model types
from actdyn.models.model import SeqVae
from actdyn.models.model_wrapper import ModelWrapper

# Common environments
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.environment.env_wrapper import GymObservationWrapper

# Common policies
from actdyn.policy.mpc import MpcICem

# Common metrics
from actdyn.metrics.information import FisherInformationMetric

__all__ = [
    # Core
    "Agent",
    "Experiment",
    "ExperimentConfig",
    "setup_experiment",
    # Models
    "SeqVae",
    "ModelWrapper",
    # Environments
    "VectorFieldEnv",
    "GymObservationWrapper",
    # Policies
    "MpcICem",
    # Metrics
    "FisherInformationMetric",
]
