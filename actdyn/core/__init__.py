"""Core components for Active Dynamics.

This package contains the high-level orchestration primitives (Agent,
Experiment). Importing :mod:`actdyn.core` is intentionally lightweight.
"""

from .agent import Agent
from .experiment import Experiment, MetaEmbeddingExperiment

__all__ = ["Agent", "Experiment", "MetaEmbeddingExperiment"]
