"""Multiple-shooting open-loop planner for exact RHC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np


@dataclass
class RhcPlanResult:
    actions: np.ndarray
    states: np.ndarray
    variances: np.ndarray
    cost: float


class RhcMultipleShootingPlanner:
    """Open-loop multiple-shooting planner following the RHC reference code."""

    def __init__(
        self,
        *,
        model,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        planner_maxiter: int = 500,
        state_bounds: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.action_low = np.asarray(action_low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float64).reshape(-1)
        self.horizon = int(max(horizon, 1))
        self.state_bounds = state_bounds
        self.planner_maxiter = int(max(planner_maxiter, 1))
        self._warm_start: np.ndarray | None = None
        self._rng = rng or np.random.default_rng(0)

    @property
    def action_dim(self) -> int:
        return int(self.action_low.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.model.output_dim)

    def plan(
        self,
        *,
        x0: np.ndarray,
        objective: str = "rhc_us",
    ) -> RhcPlanResult:
        x0_arr = np.asarray(x0, dtype=np.float64).reshape(1, self.state_dim)
        predict_cas = self.model.predict_cas()

        xu = ca.MX.sym("xu", self.horizon, self.state_dim + self.action_dim)
        states = ca.vcat((ca.DM(x0_arr), xu[:, : self.state_dim]))
        actions = xu[:, -self.action_dim :]

        lower = np.full((self.horizon, self.state_dim + self.action_dim), -np.inf, dtype=np.float64)
        upper = np.full((self.horizon, self.state_dim + self.action_dim), np.inf, dtype=np.float64)
        lower[:, -self.action_dim :] = self.action_low
        upper[:, -self.action_dim :] = self.action_high
        if self.state_bounds is not None:
            bounds = np.asarray(self.state_bounds, dtype=np.float64)
            if bounds.ndim == 2:
                bounds = np.tile(bounds[None, :, :], (self.horizon, 1, 1))
            lower[:, : self.state_dim] = bounds[:, :, 0]
            upper[:, : self.state_dim] = bounds[:, :, 1]

        objective_expr = 0
        constraints = ca.MX()
        variances = []
        feature_inputs = []
        for step in range(self.horizon):
            x_t = states[step, :]
            u_t = actions[step, :]
            xu_t = ca.hcat((x_t, u_t))
            delta_t, var_t = predict_cas(xu_t)
            variances.append(var_t)
            feature_inputs.append(xu_t)
            constraints = ca.horzcat(constraints, states[step + 1, :] - (x_t + delta_t))
            if objective == "rhc_us":
                objective_expr += -ca.sum1(var_t)

        if objective == "rhc_mvr":
            stacked_inputs = ca.vertcat(*feature_inputs)
            objective_expr = self.model.predicted_posterior_entropy_cas(stacked_inputs)
        elif objective != "rhc_us":
            raise ValueError(f"Unsupported RHC objective {objective!r}")

        flat = ca.reshape(xu, -1, 1)
        nlp = {"x": flat, "f": objective_expr, "g": constraints}
        opts = {"ipopt.max_iter": self.planner_maxiter}
        solver = ca.nlpsol("rhc_solver", "ipopt", nlp, opts)

        x0_guess = self._initial_guess(x0_arr)
        sol = solver(
            x0=x0_guess,
            lbx=lower.T.flatten(),
            ubx=upper.T.flatten(),
            lbg=0.0,
            ubg=0.0,
        )
        opt_xu = np.array(ca.reshape(sol["x"], xu.shape[0], xu.shape[1]), dtype=np.float64)
        opt_actions = opt_xu[:, -self.action_dim :]
        opt_states = np.array(ca.vcat((ca.DM(x0_arr), opt_xu[:, : self.state_dim])), dtype=np.float64)
        variance_f = ca.Function("variance", [xu], [ca.vcat(variances)])
        opt_variances = np.array(variance_f(opt_xu), dtype=np.float64).reshape(self.horizon, -1)
        cost = float(sol["f"])
        self._warm_start = opt_xu.copy()
        return RhcPlanResult(
            actions=opt_actions,
            states=opt_states,
            variances=opt_variances,
            cost=cost,
        )

    def _initial_guess(self, x0: np.ndarray) -> np.ndarray:
        if self._warm_start is not None and self._warm_start.shape[0] == self.horizon:
            warm = self._warm_start.copy()
            shifted = np.zeros_like(warm)
            shifted[:-1] = warm[1:]
            shifted[-1, -self.action_dim :] = 0.0
            shifted[:, : self.state_dim] = self._rollout_mean_states(x0, shifted[:, -self.action_dim :])[1:]
            return shifted.T.flatten()

        random_actions = self._rng.uniform(
            low=self.action_low[None, :],
            high=self.action_high[None, :],
            size=(self.horizon, self.action_dim),
        )
        mean_states = self._rollout_mean_states(x0, random_actions)
        xu0 = np.concatenate((mean_states[1:], random_actions), axis=1)
        return xu0.T.flatten()

    def _rollout_mean_states(self, x0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        states = [np.asarray(x0, dtype=np.float64).reshape(1, self.state_dim)]
        for u_t in actions:
            xu_t = np.concatenate((states[-1], u_t.reshape(1, self.action_dim)), axis=1)
            pred = self.model.predict(xu_t, return_variance=False)
            next_state = states[-1] + pred.mean
            states.append(next_state)
        return np.vstack(states)
