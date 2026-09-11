"""Sensitivity carried between observations in blockwise parameter learning."""

from collections.abc import Callable

import torch

SensitivityUpdate = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def dynamics_only_sensitivity(
    predicted_sensitivity: torch.Tensor,
    prior_precision: torch.Tensor,
    posterior_precision_chol: torch.Tensor,
) -> torch.Tensor:
    """Carry predicted sensitivity (..., state_dim, parameter_dim) unchanged."""
    return predicted_sensitivity


def measurement_corrected_sensitivity(
    predicted_sensitivity: torch.Tensor,
    prior_precision: torch.Tensor,
    posterior_precision_chol: torch.Tensor,
) -> torch.Tensor:
    """Return S+ = P+ (P-)^{-1} S- using the state filter's cached factors.

    Sensitivity has shape (..., state_dim, parameter_dim); the precision and Cholesky factor
    have shape (..., state_dim, state_dim). Output preserves the sensitivity dtype
    and shape. The observation is held fixed and state covariance derivatives
    are omitted. In particular, the Poisson term
    (dP+/dtheta_j) s_z is not included. This is a local mean-sensitivity
    approximation, not the full derivative of the nonlinear filter.
    """
    precision_sensitivity = prior_precision @ predicted_sensitivity
    return torch.cholesky_solve(precision_sensitivity, posterior_precision_chol)


LEARNING_SENSITIVITY_UPDATES: dict[str, SensitivityUpdate] = {
    "dynamics_only": dynamics_only_sensitivity,
    "measurement_corrected": measurement_corrected_sensitivity,
}


def learning_sensitivity_update(mode: str) -> SensitivityUpdate:
    """Select a sensitivity update when constructing the learner."""
    try:
        return LEARNING_SENSITIVITY_UPDATES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown learning_sensitivity={mode!r}; choose from "
            f"{tuple(LEARNING_SENSITIVITY_UPDATES)}"
        ) from None
