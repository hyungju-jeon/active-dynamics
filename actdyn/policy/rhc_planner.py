"""Official-style multiple-shooting planner for exact RHC."""

from __future__ import annotations

from dataclasses import dataclass

import casadi as cas
import numpy as np


@dataclass
class RhcPlanResult:
    actions: np.ndarray
    states: np.ndarray
    variances: np.ndarray
    cost: float


class RhcMultipleShootingPlanner:
    def __init__(
        self,
        *,
        model,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int,
        planner_maxiter: int = 500,
        state_bounds: np.ndarray | None = None,
        warm_start: bool = False,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.model = model
        self.action_low = np.asarray(action_low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float64).reshape(-1)
        self.horizon = int(max(horizon, 1))
        self.state_bounds = state_bounds
        self.planner_maxiter = int(max(planner_maxiter, 1))
        self.warm_start = bool(warm_start)
        self._warm_start: np.ndarray | None = None
        self._rng = rng or np.random.default_rng(0)
        self.model_diff = True

    @property
    def action_dim(self) -> int:
        return int(self.action_low.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.model.output_dim)

    def plan(self, *, x0: np.ndarray, objective: str = 'rhc_us') -> RhcPlanResult:
        m = self.model.predict_casf(ret_var=True)
        x0_arr = cas.DM(np.atleast_2d(np.asarray(x0, dtype=np.float64)))
        xu = cas.MX.sym('x', self.horizon, self.state_dim + self.action_dim)
        x = cas.vcat((x0_arr, xu[:, : self.state_dim]))
        u = xu[:, -self.action_dim:]

        xu_l = np.ones((self.horizon, self.state_dim + self.action_dim), dtype=np.float64) * -np.inf
        xu_u = np.ones((self.horizon, self.state_dim + self.action_dim), dtype=np.float64) * np.inf
        xu_l[:, -self.action_dim:] = self.action_low
        xu_u[:, -self.action_dim:] = self.action_high
        if self.state_bounds is not None:
            bounds = np.asarray(self.state_bounds, dtype=np.float64)
            if bounds.ndim == 2:
                bounds = np.tile(bounds[None, :, :], (self.horizon, 1, 1))
            xu_l[:, : self.state_dim] = bounds[:, :, 0]
            xu_u[:, : self.state_dim] = bounds[:, :, 1]

        obj = 0
        g = cas.MX()
        v = cas.MX()
        for i in range(self.horizon):
            xi = x[i, :]
            ui = u[i, :]
            pred_res = m(cas.hcat((xi, ui)))
            xj = pred_res[0]
            vi = pred_res[1]
            v = cas.vcat((v, vi))
            gi = x[i + 1, :] - xj
            if self.model_diff:
                gi -= xi
            g = cas.horzcat(g, gi)
            if objective == 'rhc_us':
                obj += -vi
        if objective == 'rhc_mvr':
            inputs = cas.horzcat(x[:-1, :], u)
            obj = self.model.pred_ent_cas(inputs)
        elif objective != 'rhc_us':
            raise ValueError(f'Unsupported RHC objective {objective!r}')

        xu_flat = cas.reshape(xu, -1, 1)
        nlp = {'x': xu_flat, 'f': obj, 'g': g}
        opts = {'ipopt.max_iter': self.planner_maxiter}
        solver = cas.nlpsol('solver', 'ipopt', nlp, opts)

        xopt_0 = self._initial_guess(xu_flat.shape[0])
        sol = solver(x0=xopt_0, lbx=xu_l.T.flatten(), ubx=xu_u.T.flatten(), lbg=0, ubg=0)
        opt_xu = np.array(cas.reshape(sol['x'], xu.shape[0], xu.shape[1]))
        opt_a = np.array(opt_xu[:, -self.action_dim:])
        opt_x = np.array(cas.vcat((x0_arr, opt_xu[:, : self.state_dim])))
        cost = float(sol['f'])
        if self.warm_start:
            self._warm_start = opt_xu.copy()
        var_f = cas.Function('variance', [xu], [v])
        opt_varx = np.array(var_f(opt_xu)).reshape(self.horizon, -1)
        return RhcPlanResult(actions=opt_a, states=opt_x, variances=opt_varx, cost=cost)

    def _initial_guess(self, flat_size: int) -> np.ndarray:
        if self.warm_start and self._warm_start is not None:
            return cas.reshape(self._warm_start, -1, 1)
        return (self._rng.random(flat_size) - 0.5) * 4.0
