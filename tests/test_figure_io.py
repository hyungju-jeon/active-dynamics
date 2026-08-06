from __future__ import annotations

import math

import numpy as np
import pytest

from actdyn.utils.figure_io import (
    centered_moving_average,
    finite_mean,
    finite_median,
    finite_quantile,
)


def test_finite_stats_ignore_nan_and_empty_values() -> None:
    values = [1.0, float("nan"), 3.0, float("inf")]

    assert finite_mean(values) == pytest.approx(2.0)
    assert finite_median(values) == pytest.approx(2.0)
    assert finite_quantile(values, 0.75) == pytest.approx(2.5)
    assert math.isnan(finite_mean([float("nan")]))


def test_centered_moving_average_keeps_short_inputs() -> None:
    values = np.asarray([1.0, 2.0], dtype=np.float64)

    assert np.array_equal(centered_moving_average(values, width=3), values)
    assert np.array_equal(centered_moving_average(values, width=0), values)
