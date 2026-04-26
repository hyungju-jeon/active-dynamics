"""Policy-agnostic surrogate models for planning and active learning.

These surrogates are intentionally placed in the models layer rather than under
`actdyn.policy` so they can be reused by multiple planners/policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import casadi as cas
import numpy as np
import scipy.optimize as sopt


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


@dataclass
class PredictiveMoments:
    mean: np.ndarray
    variance: np.ndarray | None = None


RHCObjective = Literal["us", "mvr", "rhc_us", "rhc_mvr"]


class RFFBayesianLinearDynamics:
    """Bayesian linear regression with sinusoidal random Fourier features."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        num_features: int = 64,
        bandwidth_init: float | np.ndarray = 1.0,
        beta: float = 1.0,
        prior_precision: float = 1e-8,
        seed: int = 0,
        optimize_hyperparams: bool = True,
        opt_maxsteps: int = 150,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_features = int(num_features)
        self.beta = float(max(beta, 1e-12))
        self.prior_precision = float(max(prior_precision, 1e-16))
        self.optimize_hyperparams = bool(optimize_hyperparams)
        self.opt_maxsteps = int(max(opt_maxsteps, 1))
        self._rng = np.random.default_rng(int(seed))

        if np.isscalar(bandwidth_init) or np.size(bandwidth_init) == 1:
            bw = float(np.asarray(bandwidth_init).reshape(-1)[0])
            self.v = np.repeat(max(bw, 1e-6), self.input_dim).astype(np.float64)
        else:
            v = np.asarray(bandwidth_init, dtype=np.float64).reshape(-1)
            if v.shape[0] != self.input_dim:
                raise ValueError(
                    f"Expected bandwidth_init with {self.input_dim} entries, got {v.shape[0]}"
                )
            self.v = np.clip(v, 1e-6, None)

        self.mean0 = np.zeros((self.num_features, self.output_dim), dtype=np.float64)
        self.precision0 = np.eye(self.num_features, dtype=np.float64) * self.prior_precision
        self.covariance0 = np.linalg.inv(self.precision0)

        self.mean = self.mean0.copy()
        self.precision = self.precision0.copy()
        self.covariance = self.covariance0.copy()

        self._X = np.zeros((0, self.input_dim), dtype=np.float64)
        self._Y = np.zeros((0, self.output_dim), dtype=np.float64)
        self.W: np.ndarray | None = None
        self.psi: np.ndarray | None = None
        self.reset()

    @property
    def num_samples(self) -> int:
        return int(self._X.shape[0])

    @property
    def num_points(self) -> int:
        return self.num_samples

    @property
    def lengthscale(self) -> float:
        return float(np.mean(self.v))

    def reset(self) -> None:
        self.mean = self.mean0.copy()
        self.precision = self.precision0.copy()
        self.covariance = self.covariance0.copy()
        self.W = self._rng.normal(0.0, 1.0, size=(self.num_features, self.input_dim))
        self.psi = self._rng.uniform(0.0, 2.0 * np.pi, size=(self.num_features,))
        if self.num_features > 0:
            self.W[0, :] = 0.0
            self.psi[0] = 0.0

    def add_episode(self, inputs: np.ndarray, deltas: np.ndarray) -> None:
        x = _as_2d(inputs)
        y = _as_2d(deltas)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Expected matching inputs/targets, got {x.shape} and {y.shape}")
        if x.shape[1] != self.input_dim or y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected input/output dims {(self.input_dim, self.output_dim)}, got {(x.shape[1], y.shape[1])}"
            )
        self._X = np.vstack((self._X, x))
        self._Y = np.vstack((self._Y, y))
        if self.optimize_hyperparams:
            self.retrain(opt_hyperparams=True)
        else:
            self.update_param_dist(x, y)

    def retrain(self, opt_hyperparams: bool = False) -> None:
        if opt_hyperparams and self.num_samples > 0:
            self.opt_hyperparams()
        self.mean = self.mean0.copy()
        self.precision = self.precision0.copy()
        self.covariance = self.covariance0.copy()
        if self.num_samples > 0:
            self.update_param_dist(self._X, self._Y)

    def update_param_dist(self, x: np.ndarray, y: np.ndarray) -> None:
        x = _as_2d(x)
        y = _as_2d(y)
        phi_x = self.phi(x)
        precision_new = self.precision + self.beta * np.dot(phi_x.T, phi_x)
        mean_new = np.linalg.solve(
            precision_new,
            np.dot(self.precision, self.mean) + self.beta * np.dot(phi_x.T, y),
        )
        self.mean = mean_new
        self.precision = 0.5 * (precision_new + precision_new.T)
        self.covariance = np.linalg.inv(self.precision)

    def phi(self, x: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        if self.W is None or self.psi is None:
            raise RuntimeError("RFF model not initialized")
        if v is None:
            v = self.v
        x_arr = _as_2d(x)
        if x_arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x_arr.shape[1]}")
        y = np.sin((self.W / v[None, :]) @ x_arr.T + self.psi[:, None])
        return y.T

    def phi_cas(self, x: cas.MX | cas.SX, v: np.ndarray | cas.MX | None = None):
        if self.W is None or self.psi is None:
            raise RuntimeError("RFF model not initialized")
        if v is None:
            v_mat = cas.DM(self.v[None, :])
        else:
            v_arr = np.asarray(v, dtype=np.float64).reshape(1, -1)
            v_mat = cas.DM(v_arr)
        y = cas.sin(cas.mtimes(cas.DM(self.W) / cas.repmat(v_mat, self.W.shape[0], 1), x.T) + cas.DM(self.psi))
        return y.T

    def predict(self, x: np.ndarray, ret_var: bool = False, return_variance: bool | None = None):
        if return_variance is not None:
            ret_var = bool(return_variance)
        phi = self.phi(x)
        mu = np.dot(self.mean.T, phi.T).T
        if ret_var:
            sigma = 1.0 / self.beta + np.sum(phi * (self.covariance @ phi.T).T, axis=-1)
            return PredictiveMoments(mean=mu, variance=sigma.reshape(-1, 1))
        return PredictiveMoments(mean=mu, variance=None)

    def predict_casf(self, ret_var: bool = False):
        x = cas.MX.sym("x", 1, self.input_dim)
        phi = self.phi_cas(x)
        mu = cas.mtimes(cas.DM(self.mean).T, phi.T).T
        res = [mu]
        if ret_var:
            sigma = 1.0 / self.beta + cas.sum2(phi * cas.mtimes(cas.DM(self.covariance), phi.T).T)
            res.append(sigma)
        return cas.Function("f_mu", [x], res)

    def pred_ent_cas(self, x: cas.MX | cas.SX):
        phi_x = self.phi_cas(x)
        precision_new = cas.DM(self.precision) + self.beta * cas.mtimes(phi_x.T, phi_x)
        log_det_f = self.log_det_cas(int(precision_new.shape[0]))
        logdet_cov = -log_det_f(precision_new)
        k = self.num_features
        return k / 2 + k / 2 * cas.log(2 * cas.pi) + logdet_cov / 2

    def predicted_posterior_entropy_cas(self, inputs: cas.MX | cas.SX):
        return self.pred_ent_cas(inputs)

    def log_det_cas(self, size: int):
        S = cas.SX.sym("s", size, size)
        return cas.Function('log_det', [S], [cas.trace(cas.log(cas.qr(S)[1]))]).expand()

    def opt_hyperparams(self):
        if self.num_samples == 0:
            return 0.0

        def nllh(v_flat: np.ndarray) -> float:
            v = np.clip(np.asarray(v_flat, dtype=np.float64).reshape(-1), 1e-6, None)
            return -float(self.llh(v=v))

        res = sopt.minimize(nllh, x0=self.v, method='Nelder-Mead', options={'maxiter': self.opt_maxsteps})
        if res.success:
            self.v = np.clip(np.asarray(res.x, dtype=np.float64).reshape(-1), 1e-6, None)
            return float(res.fun)
        return float(nllh(self.v))

    def llh(
        self,
        v: np.ndarray | None = None,
        x_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
    ) -> float:
        if (x_eval is None) != (y_eval is None):
            raise ValueError("x_eval and y_eval must either both be set or both be None")
        if x_eval is None:
            x_eval = self._X
            y_eval = self._Y
        x_eval = _as_2d(x_eval)
        y_eval = _as_2d(y_eval)
        if x_eval.shape[0] == 0:
            return 0.0
        if v is not None:
            phi_x = self.phi(self._X, np.asarray(v, dtype=np.float64).reshape(-1))
            precision = self.precision0 + self.beta * phi_x.T @ phi_x
            mean = np.linalg.solve(precision, self.precision0 @ self.mean0 + self.beta * (phi_x.T @ self._Y))
            phi_eval = self.phi(x_eval, np.asarray(v, dtype=np.float64).reshape(-1))
            cov = np.linalg.inv(precision)
        else:
            phi_eval = self.phi(x_eval)
            mean = self.mean
            cov = self.covariance
        y_pred = (mean.T @ phi_eval.T).T
        sigma = 1.0 / self.beta + np.sum(phi_eval * (cov @ phi_eval.T).T, axis=1)
        y_diff = y_eval - y_pred
        llht = np.sum(y_diff ** 2, axis=1) / sigma
        return -float(np.sum(llht))


class LocalRBFBayesianLinearDynamics:
    """Bayesian linear regression with fixed localized RBF bases on a grid."""

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        input_low: np.ndarray,
        input_high: np.ndarray,
        grid_points: int = 3,
        epsilon: float = 0.01,
        beta: float = 1.0,
        prior_precision: float = 1e-8,
        include_bias: bool = False,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.beta = float(max(beta, 1e-12))
        self.prior_precision = float(max(prior_precision, 1e-16))
        self.grid_points = int(max(grid_points, 1))
        self.epsilon = float(np.clip(epsilon, 1e-8, 0.5))
        self.include_bias = bool(include_bias)

        self.input_low = np.asarray(input_low, dtype=np.float64).reshape(-1)
        self.input_high = np.asarray(input_high, dtype=np.float64).reshape(-1)
        if self.input_low.shape[0] != self.input_dim or self.input_high.shape[0] != self.input_dim:
            raise ValueError(
                f"Expected input bounds with {self.input_dim} entries, got "
                f"{self.input_low.shape[0]} and {self.input_high.shape[0]}"
            )
        if np.any(self.input_high <= self.input_low):
            raise ValueError("input_high must be strictly greater than input_low for all dimensions")

        self._centers, self._lengthscale_vec = self._build_grid()
        self.num_features = int(self._centers.shape[0]) + (1 if self.include_bias else 0)

        self.mean0 = np.zeros((self.num_features, self.output_dim), dtype=np.float64)
        self.precision0 = np.eye(self.num_features, dtype=np.float64) * self.prior_precision
        self.covariance0 = np.linalg.inv(self.precision0)

        self.mean = self.mean0.copy()
        self.precision = self.precision0.copy()
        self.covariance = self.covariance0.copy()
        self._X = np.zeros((0, self.input_dim), dtype=np.float64)
        self._Y = np.zeros((0, self.output_dim), dtype=np.float64)

    @property
    def num_samples(self) -> int:
        return int(self._X.shape[0])

    @property
    def num_points(self) -> int:
        return self.num_samples

    @property
    def lengthscale(self) -> float:
        return float(np.mean(self._lengthscale_vec))

    def _build_grid(self) -> tuple[np.ndarray, np.ndarray]:
        axes: list[np.ndarray] = []
        spacings: list[float] = []
        for low, high in zip(self.input_low, self.input_high, strict=True):
            axis = np.linspace(low, high, self.grid_points, dtype=np.float64)
            axes.append(axis)
            if self.grid_points > 1:
                spacing = float(axis[1] - axis[0])
            else:
                spacing = float(max(high - low, 1.0))
            spacings.append(max(spacing, 1e-6))
        mesh = np.meshgrid(*axes, indexing="ij")
        centers = np.stack([m.reshape(-1) for m in mesh], axis=1)
        denom = np.sqrt(2.0 * np.log(1.0 / self.epsilon))
        lengthscale_vec = np.asarray(spacings, dtype=np.float64) / max(denom, 1e-6)
        return centers, np.clip(lengthscale_vec, 1e-6, None)

    def add_episode(self, inputs: np.ndarray, deltas: np.ndarray) -> None:
        x = _as_2d(inputs)
        y = _as_2d(deltas)
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Expected matching inputs/targets, got {x.shape} and {y.shape}")
        if x.shape[1] != self.input_dim or y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected input/output dims {(self.input_dim, self.output_dim)}, got {(x.shape[1], y.shape[1])}"
            )
        self._X = np.vstack((self._X, x))
        self._Y = np.vstack((self._Y, y))
        self.update_param_dist(x, y)

    def update_param_dist(self, x: np.ndarray, y: np.ndarray) -> None:
        x = _as_2d(x)
        y = _as_2d(y)
        phi_x = self.phi(x)
        precision_new = self.precision + self.beta * np.dot(phi_x.T, phi_x)
        mean_new = np.linalg.solve(
            precision_new,
            np.dot(self.precision, self.mean) + self.beta * np.dot(phi_x.T, y),
        )
        self.mean = mean_new
        self.precision = 0.5 * (precision_new + precision_new.T)
        self.covariance = np.linalg.inv(self.precision)

    def phi(self, x: np.ndarray) -> np.ndarray:
        x_arr = _as_2d(x)
        if x_arr.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {x_arr.shape[1]}")
        diffs = (x_arr[:, None, :] - self._centers[None, :, :]) / self._lengthscale_vec[None, None, :]
        features = np.exp(-0.5 * np.sum(diffs ** 2, axis=-1))
        if self.include_bias:
            features = np.concatenate(
                [np.ones((features.shape[0], 1), dtype=np.float64), features],
                axis=1,
            )
        return features

    def phi_cas(self, x: cas.MX | cas.SX):
        lengthscale = cas.DM(self._lengthscale_vec.reshape(1, -1))
        num_rows = x.shape[0]
        cols = []
        for center in self._centers:
            center_row = cas.DM(center.reshape(1, -1))
            diffs = (x - cas.repmat(center_row, num_rows, 1)) / cas.repmat(lengthscale, num_rows, 1)
            sq_norm = cas.sum2(diffs ** 2)
            cols.append(cas.exp(-0.5 * sq_norm))
        features = cas.hcat(cols)
        if self.include_bias:
            features = cas.hcat((cas.DM.ones(num_rows, 1), features))
        return features

    def predict(self, x: np.ndarray, ret_var: bool = False, return_variance: bool | None = None):
        if return_variance is not None:
            ret_var = bool(return_variance)
        phi = self.phi(x)
        mu = np.dot(self.mean.T, phi.T).T
        if ret_var:
            sigma = 1.0 / self.beta + np.sum(phi * (self.covariance @ phi.T).T, axis=-1)
            return PredictiveMoments(mean=mu, variance=sigma.reshape(-1, 1))
        return PredictiveMoments(mean=mu, variance=None)

    def predict_casf(self, ret_var: bool = False):
        x = cas.MX.sym("x", 1, self.input_dim)
        phi = self.phi_cas(x)
        mu = cas.mtimes(cas.DM(self.mean).T, phi.T).T
        res = [mu]
        if ret_var:
            sigma = 1.0 / self.beta + cas.sum2(phi * cas.mtimes(cas.DM(self.covariance), phi.T).T)
            res.append(sigma)
        return cas.Function("f_mu", [x], res)

    def pred_ent_cas(self, x: cas.MX | cas.SX):
        phi_x = self.phi_cas(x)
        precision_new = cas.DM(self.precision) + self.beta * cas.mtimes(phi_x.T, phi_x)
        log_det_f = self.log_det_cas(int(precision_new.shape[0]))
        logdet_cov = -log_det_f(precision_new)
        k = self.num_features
        return k / 2 + k / 2 * cas.log(2 * cas.pi) + logdet_cov / 2

    def predicted_posterior_entropy_cas(self, inputs: cas.MX | cas.SX):
        return self.pred_ent_cas(inputs)

    def log_det_cas(self, size: int):
        S = cas.SX.sym("s", size, size)
        return cas.Function("log_det", [S], [cas.trace(cas.log(cas.qr(S)[1]))]).expand()
