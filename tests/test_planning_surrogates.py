from __future__ import annotations

import numpy as np
import pytest

from actdyn.models.planning_surrogates import (
    LocalRBFBayesianLinearDynamics,
    RFFBayesianLinearDynamics,
)


@pytest.mark.parametrize(
    "model",
    [
        RFFBayesianLinearDynamics(
            input_dim=2,
            output_dim=1,
            num_features=4,
            optimize_hyperparams=False,
            seed=0,
        ),
        LocalRBFBayesianLinearDynamics(
            input_dim=2,
            output_dim=1,
            input_low=np.array([-1.0, -1.0]),
            input_high=np.array([1.0, 1.0]),
            grid_points=2,
        ),
    ],
)
def test_surrogate_update_predicts_moments(model) -> None:
    x = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float64)
    y = np.array([[0.0], [1.0]], dtype=np.float64)

    model.add_episode(x, y)
    moments = model.predict(x, ret_var=True)

    assert model.num_samples == 2
    assert model.num_points == 2
    assert moments.mean.shape == (2, 1)
    assert moments.variance is not None
    assert moments.variance.shape == (2, 1)
