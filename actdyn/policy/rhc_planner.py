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
        action_radius: float | None = None,
    ) -> None:
        """``action_radius``: when set, each planned input also satisfies ||u_k|| <= action_radius
        (ball constraint) besides the box [action_low, action_high]; None keeps the box only."""
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
        self.action_radius = None if action_radius is None else float(action_radius)

    @property
    def action_dim(self) -> int:
        return int(self.action_low.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.model.output_dim)

    def plan(
        self, *, x0: np.ndarray, objective: str = 'rhc_us', input_offset: np.ndarray | None = None,
        action_scale: float = 1.0,
    ) -> RhcPlanResult:
        """Plan ``horizon`` actions from ``x0``.

        ``input_offset`` (horizon, action_dim) is a known input added to each planned
        action before the surrogate sees it (decision-session evidence); the returned
        actions exclude it. ``action_scale`` multiplies the box and the ball radius of
        this plan (the session's bound, ``SessionRule.action_scale``).
        """
        action_low = float(action_scale) * self.action_low
        action_high = float(action_scale) * self.action_high
        action_radius = None if self.action_radius is None else float(action_scale) * self.action_radius
        m = self.model.predict_casf(ret_var=True)
        offset = cas.DM(np.zeros((self.horizon, self.action_dim)) if input_offset is None
                        else np.asarray(input_offset, dtype=np.float64).reshape(self.horizon, self.action_dim))
        x0_arr = cas.DM(np.atleast_2d(np.asarray(x0, dtype=np.float64)))
        xu = cas.MX.sym('x', self.horizon, self.state_dim + self.action_dim)
        x = cas.vcat((x0_arr, xu[:, : self.state_dim]))
        u = xu[:, -self.action_dim:]

        xu_l = np.ones((self.horizon, self.state_dim + self.action_dim), dtype=np.float64) * -np.inf
        xu_u = np.ones((self.horizon, self.state_dim + self.action_dim), dtype=np.float64) * np.inf
        xu_l[:, -self.action_dim:] = action_low
        xu_u[:, -self.action_dim:] = action_high
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
            pred_res = m(cas.hcat((xi, ui + offset[i, :])))
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

        lbg, ubg = 0, 0
        if action_radius is not None:
            # Ball constraint: sum_j u_kj^2 <= r^2 for every planned step, after the dynamics.
            n_dynamics = int(g.numel())
            g = cas.horzcat(g, cas.horzcat(*[cas.sumsqr(u[i, :]) for i in range(self.horizon)]))
            lbg = np.concatenate((np.zeros(n_dynamics), np.full(self.horizon, -np.inf)))
            ubg = np.concatenate((np.zeros(n_dynamics), np.full(self.horizon, action_radius ** 2)))

        xu_flat = cas.reshape(xu, -1, 1)
        nlp = {'x': xu_flat, 'f': obj, 'g': g}
        opts = {'ipopt.max_iter': self.planner_maxiter}
        solver = cas.nlpsol('solver', 'ipopt', nlp, opts)

        xopt_0 = self._initial_guess(xu_flat.shape[0])
        sol = solver(x0=xopt_0, lbx=xu_l.T.flatten(), ubx=xu_u.T.flatten(), lbg=lbg, ubg=ubg)
        opt_xu = np.array(cas.reshape(sol['x'], xu.shape[0], xu.shape[1]))
        opt_a = np.array(opt_xu[:, -self.action_dim:])
        if action_radius is not None:
            # Remove the solver's constraint tolerance so executed inputs lie in the ball.
            norms = np.linalg.norm(opt_a, axis=1, keepdims=True)
            opt_a = opt_a * np.minimum(1.0, action_radius / np.maximum(norms, 1e-12))
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
