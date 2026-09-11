"""Measurement steps for local Gaussian parameter-information rollouts."""

from collections.abc import Callable

import torch

from actdyn.utils.torch_utils import safe_cholesky, symmetrize

PLANNING_ROLLOUT_REVISION = "all_objectives_v2"

MeasurementUpdate = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def prediction_only_measurement_update(
    prior_cov: torch.Tensor,
    prior_sensitivity: torch.Tensor,
    state_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Carry predictive covariance (..., d, d) and nominal sensitivity (..., d, p)."""
    return prior_cov, prior_sensitivity


def conditioned_measurement_update(
    prior_cov: torch.Tensor,
    prior_sensitivity: torch.Tensor,
    state_info: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Condition covariance and mean sensitivity on a fixed nominal observation.

    Covariance and state information have shape (..., d, d); sensitivity has
    shape (..., d, p). With frozen local Gaussian matrices, this computes
    P+ = (P-^{-1} + I_z)^{-1} and S+ = P+ P-^{-1} S- = (I-KC) S-.
    The observation is held fixed when differentiating, even at zero innovation.
    Gain/curvature derivatives and covariance contributions to Fisher information
    are omitted. No state mean or parameter belief is updated.
    """
    prior_precision = torch.cholesky_inverse(safe_cholesky(symmetrize(prior_cov)))
    posterior_chol = safe_cholesky(symmetrize(prior_precision + state_info))
    posterior_cov = symmetrize(torch.cholesky_inverse(posterior_chol))
    # Solve in precision form to avoid subtracting nearly equal sensitivities.
    posterior_sensitivity = torch.cholesky_solve(
        prior_precision @ prior_sensitivity, posterior_chol
    )
    return posterior_cov, posterior_sensitivity


PLANNING_MEASUREMENT_UPDATES: dict[str, MeasurementUpdate] = {
    "prediction_only": prediction_only_measurement_update,
    "measurement_conditioned": conditioned_measurement_update,
}


def planning_measurement_update(mode: str) -> MeasurementUpdate:
    """Select the measurement step once, when constructing a planning objective."""
    try:
        return PLANNING_MEASUREMENT_UPDATES[mode]
    except KeyError:
        raise ValueError(
            f"Unknown planning_rollout={mode!r}; choose from "
            f"{tuple(PLANNING_MEASUREMENT_UPDATES)}"
        ) from None
