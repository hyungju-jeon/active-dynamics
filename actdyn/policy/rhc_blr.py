"""Bayesian RFF dynamics model used by the exact RHC baseline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import casadi as cs
import numpy as np
from scipy.optimize import minimize_scalar


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


@dataclass
class BLRPosterior:
    mean_w: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    beta: float
    alpha: float
    lengthscale: float


@dataclass
class PredictiveMoments:
    mean: np.ndarray
    variance: np.ndarray | None = None


class RFFBayesianLinearDynamics:
    """Bayesian linear regression on random Fourier features.

    The model fits discrete state differences ``delta_x = x_{t+1} - x_t`` from
    concatenated state-action inputs. It keeps the full Gaussian posterior over
    the linear weights, which is what RHC needs for predictive-variance and
    model-entropy objectives.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_features: int = 64,
        prior_precision: float = 1.0,
        obs_noise_var: float = 1e-3,
        lengthscale: float = 1.0,
        seed: int = 0,
        optimize_lengthscale: bool = True,
        lengthscale_bounds: tuple[float, float] = (0.1, 10.0),
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_features = int(num_features)
        self.prior_precision = float(prior_precision)
        self.obs_noise_var = float(max(obs_noise_var, 1e-8))
        self.beta = 1.0 / self.obs_noise_var
        self.lengthscale = float(max(lengthscale, 1e-3))
        self.optimize_lengthscale = bool(optimize_lengthscale)
        self.lengthscale_bounds = (
            float(lengthscale_bounds[0]),
            float(lengthscale_bounds[1]),
        )
        self._rng = np.random.default_rng(int(seed))
        self._base_freq = self._rng.normal(size=(self.num_features, self.input_dim))
        self._phase = self._rng.uniform(0.0, 2.0 * math.pi, size=(self.num_features,))
        self._X = np.zeros((0, self.input_dim), dtype=np.float64)
        self._Y = np.zeros((0, self.output_dim), dtype=np.float64)
        self.posterior = BLRPosterior(
            mean_w=np.zeros((self.num_features, self.output_dim), dtype=np.float64),
            covariance=(1.0 / self.prior_precision) * np.eye(self.num_features, dtype=np.float64),
            precision=self.prior_precision * np.eye(self.num_features, dtype=np.float64),
            beta=self.beta,
            alpha=self.prior_precision,
            lengthscale=self.lengthscale,
        )

    @property
    def num_samples(self) -> int:
        return int(self._X.shape[0])

    @property
    def num_points(self) -> int:
        return self.num_samples

    def add_episode(self, inputs: np.ndarray, deltas: np.ndarray) -> None:
        x = _as_2d(inputs)
        y = _as_2d(deltas)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Expected matching inputs/targets, got {x.shape} and {y.shape}")
        if x.shape[1] != self.input_dim or y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected input/output dims {(self.input_dim, self.output_dim)},"
                f" got {(x.shape[1], y.shape[1])}"
            )
        self._X = np.concatenate([self._X, x], axis=0)
        self._Y = np.concatenate([self._Y, y], axis=0)
        self.refit()

    def refit(self) -> None:
        if self.num_samples == 0:
            return
        if self.optimize_lengthscale and self.num_samples >= max(6, self.input_dim + 2):
            self.lengthscale = self._optimize_lengthscale()
        phi = self._feature_matrix(self._X, self.lengthscale)
        self.posterior = self._posterior_from_features(phi, self._Y, self.lengthscale)

    def predict_delta(self, inputs: np.ndarray) -> np.ndarray:
        x = _as_2d(inputs)
        phi = self._feature_matrix(x, self.lengthscale)
        return phi @ self.posterior.mean_w

    def predict(
        self,
        inputs: np.ndarray,
        *,
        return_variance: bool = True,
    ) -> PredictiveMoments:
        x = _as_2d(inputs)
        mean = self.predict_delta(x)
        variance = None
        if return_variance:
            var_scalar = np.asarray(
                [self.predictive_variance_scalar(row) for row in x],
                dtype=np.float64,
            ).reshape(-1, 1)
            variance = np.repeat(var_scalar, self.output_dim, axis=1)
        return PredictiveMoments(mean=mean, variance=variance)

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        xu = np.concatenate([np.asarray(state, dtype=np.float64), np.asarray(action, dtype=np.float64)])
        delta = self.predict_delta(xu)[0]
        return np.asarray(state, dtype=np.float64) + delta

    def predictive_variance_scalar(self, inputs: np.ndarray) -> float:
        x = _as_2d(inputs)
        phi = self._feature_matrix(x, self.lengthscale)
        var = float(phi @ self.posterior.covariance @ phi.T)
        return float(max(var, 0.0))

    def entropy_after_inputs(self, inputs: np.ndarray) -> float:
        x = _as_2d(inputs)
        phi = self._feature_matrix(x, self.lengthscale)
        precision = self.posterior.precision + self.beta * (phi.T @ phi)
        sign, logdet = np.linalg.slogdet(precision)
        if sign <= 0:
            return float("inf")
        # Differential entropy of the Gaussian weight posterior up to constants.
        return float(-0.5 * self.output_dim * logdet)

    def casadi_feature(self, xu: cs.MX | cs.SX) -> cs.MX:
        w = cs.DM(self._base_freq / self.lengthscale)
        phase = cs.DM(self._phase.reshape(-1, 1))
        feat = math.sqrt(2.0 / self.num_features) * cs.cos(w @ xu + phase)
        return feat

    def casadi_next_state(self, state: cs.MX | cs.SX, action: cs.MX | cs.SX) -> cs.MX:
        xu = cs.vertcat(state, action)
        phi = self.casadi_feature(xu)
        mean_w = cs.DM(self.posterior.mean_w)
        delta = mean_w.T @ phi
        return state + delta

    def casadi_predictive_variance(self, state: cs.MX | cs.SX, action: cs.MX | cs.SX) -> cs.MX:
        xu = cs.vertcat(state, action)
        phi = self.casadi_feature(xu)
        cov = cs.DM(self.posterior.covariance)
        return cs.mtimes([phi.T, cov, phi])

    def casadi_posterior_entropy_cost(
        self,
        feature_list: list[cs.MX | cs.SX],
    ) -> cs.MX:
        precision = cs.DM(self.posterior.precision)
        ident = cs.DM(np.eye(self.num_features, dtype=np.float64))
        for feat in feature_list:
            precision = precision + self.beta * cs.mtimes(feat, feat.T)
        precision = precision + 1e-9 * ident
        return -cs.log(cs.det(precision))

    def predict_cas(self):
        def _predict(xu: cs.MX | cs.SX):
            phi = self.casadi_feature(xu.T if xu.shape[0] == 1 else xu)
            mean_w = cs.DM(self.posterior.mean_w)
            cov = cs.DM(self.posterior.covariance)
            delta = (mean_w.T @ phi).T
            var_scalar = cs.mtimes([phi.T, cov, phi])
            var_vec = cs.repmat(var_scalar, self.output_dim, 1)
            return delta, var_vec

        return _predict

    def predicted_posterior_entropy_cas(self, inputs: cs.MX | cs.SX) -> cs.MX:
        feature_list = [self.casadi_feature(inputs[i, :].T) for i in range(int(inputs.shape[0]))]
        return self.casadi_posterior_entropy_cost(feature_list)

    def _posterior_from_features(
        self,
        phi: np.ndarray,
        targets: np.ndarray,
        lengthscale: float,
    ) -> BLRPosterior:
        ident = np.eye(self.num_features, dtype=np.float64)
        precision = self.prior_precision * ident + self.beta * (phi.T @ phi)
        precision = 0.5 * (precision + precision.T)
        covariance = np.linalg.pinv(precision)
        mean_w = self.beta * covariance @ phi.T @ targets
        return BLRPosterior(
            mean_w=mean_w,
            covariance=covariance,
            precision=precision,
            beta=self.beta,
            alpha=self.prior_precision,
            lengthscale=lengthscale,
        )

    def _feature_matrix(self, x: np.ndarray, lengthscale: float) -> np.ndarray:
        inputs = _as_2d(x)
        scaled = inputs @ (self._base_freq / float(max(lengthscale, 1e-6))).T
        feat = math.sqrt(2.0 / self.num_features) * np.cos(scaled + self._phase[None, :])
        return feat.astype(np.float64, copy=False)

    def _optimize_lengthscale(self) -> float:
        lo, hi = self.lengthscale_bounds
        lo = max(lo, 1e-3)
        hi = max(hi, lo + 1e-3)

        def objective(log_lengthscale: float) -> float:
            lengthscale = float(np.exp(log_lengthscale))
            phi = self._feature_matrix(self._X, lengthscale)
            ident_n = np.eye(self.num_samples, dtype=np.float64)
            cov_y = (
                (1.0 / self.beta) * ident_n
                + (1.0 / self.prior_precision) * (phi @ phi.T)
            )
            cov_y = 0.5 * (cov_y + cov_y.T) + 1e-9 * ident_n
            sign, logdet = np.linalg.slogdet(cov_y)
            if sign <= 0:
                return float("inf")
            inv_cov_y = np.linalg.pinv(cov_y)
            quad = float(np.trace(self._Y.T @ inv_cov_y @ self._Y))
            return 0.5 * (
                self.output_dim * logdet
                + quad
                + self.num_samples * self.output_dim * math.log(2.0 * math.pi)
            )

        result = minimize_scalar(
            objective,
            bounds=(math.log(lo), math.log(hi)),
            method="bounded",
            options={"xatol": 1e-2, "maxiter": 32},
        )
        if not result.success:
            return self.lengthscale
        return float(np.exp(result.x))


RHCObjective = Literal["us", "mvr"]
