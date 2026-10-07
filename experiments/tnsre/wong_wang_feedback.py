"""Tracking MPC with input corrections between replanning steps.

The policy is u_t = project(v_t + g (r_t - (z_1-z_0)) d), d=(-1,1)/sqrt(2).
The planner optimizes feedforward inputs v under this feedback policy. Its
rollouts use simulated states; execution uses only the spike-count EKF state.
Future observation errors are not simulated. This is an explicit approximation,
not access to the SNN state. All learned methods use the same controller.
"""
from __future__ import annotations

import numpy as np
import torch

from actdyn.utils.validation import optimize_icem
from experiments.tnsre import eval_spiking_sessions as ev
from experiments.tnsre.wong_wang_control import TrackingController


def feedback_input(feedforward, state, reference, remaining, *, gain, bound, dt):
    """Apply gap feedback and project onto instantaneous and remaining-energy bounds.

    Float tensors feedforward/state have shape (...,2); remaining broadcasts to
    (...). Coordinates place the target in component 1. Return the same dtype
    and shape as state. The action uses the state before the current step.
    """
    direction = state.new_tensor([-1., 1.])/np.sqrt(2.)
    error = reference-(state[..., 1]-state[..., 0])
    action = feedforward+gain*error[..., None]*direction
    remaining = torch.as_tensor(remaining, dtype=state.dtype, device=state.device)
    cap = torch.sqrt(torch.clamp(remaining, min=0.)/dt).clamp(max=bound)
    norm = torch.linalg.vector_norm(action, dim=-1)
    scale = torch.minimum(torch.ones_like(norm), cap/norm.clamp_min(torch.finfo(state.dtype).tiny))
    return action*scale[..., None]


def feedback_rollout_statistics(transition, z0, drive, feedforward, reference, weights,
                                *, dt, budget, bound, gain, seed, samples=8, noise_var=.1):
    """Score state-feedback policies under shared model-process noise draws.

    z0 (2,), drive (T,2), feedforward (K,T,2), reference/weights (T,).
    Return MSE (K,), spent energy (K,samples), and peak norm (K,samples).
    Controls depend on past simulated noise and obey bounds in every trajectory.
    No future noise or future SNN observations enter an action.
    """
    if not np.isfinite([dt, budget, bound, gain, noise_var]).all() or (
            dt <= 0 or budget < 0 or bound <= 0 or gain < 0 or noise_var < 0 or samples < 1):
        raise ValueError('Invalid feedback rollout settings')
    w, v, initial, r, weight = [torch.as_tensor(np.asarray(x), dtype=torch.float32)
        for x in (drive, feedforward, z0, reference, weights)]
    if (w.ndim != 2 or w.shape[-1] != 2 or v.ndim != 3 or v.shape[1:] != w.shape
            or initial.shape != (2,) or r.shape != (len(w),) or weight.shape != r.shape
            or torch.any(weight < 0) or weight.sum() <= 0):
        raise ValueError('Invalid feedback rollout shapes or scoring weights')
    if not all(torch.isfinite(x).all() for x in (w, v, initial, r, weight)):
        raise ValueError('Feedback rollout inputs must be finite')
    rng = torch.Generator(device='cpu').manual_seed(int(seed))
    noise = torch.randn(len(w), samples, 2, generator=rng)*np.sqrt(noise_var*dt)
    z = initial.expand(len(v), samples, 2).clone()
    remaining = torch.full((len(v), samples), float(budget))
    mse, peak = torch.zeros_like(remaining), torch.zeros_like(remaining)
    with torch.no_grad():
        for t in range(len(w)):
            u = feedback_input(v[:, t, None, :], z, r[t], remaining,
                               gain=gain, bound=bound, dt=dt)
            power = torch.sum(u*u, dim=-1)
            remaining = torch.clamp(remaining-dt*power, min=0.)
            peak = torch.maximum(peak, torch.sqrt(power))
            inp = u+w[t]
            z = transition(z.reshape(-1, 2), inp.reshape(-1, 2)).reshape_as(z)+noise[t]
            mse += weight[t]*(z[..., 1]-z[..., 0]-r[t])**2
    return dict(mse=(mse.mean(-1)/weight.sum()).numpy(),
                energy=(budget-remaining).numpy(), peak=peak.numpy())


def feedback_tracking_plan(transition, z0, drive, reference, weights, *, task, dt,
                           budget, gain, seed, action_seed, seed_candidates=None, settings=None):
    """Optimize nominal inputs under feedback; actual inputs are projected per bin."""
    settings = {} if settings is None else dict(settings)
    samples = settings.pop('mc_samples', int(ev.PROTOCOL['planner_mc_samples']))
    noise_var = settings.pop('noise_var', float(ev.PROTOCOL['planner_noise_var']))

    def score(coarse):
        nominal = coarse[0].repeat_interleave(task.coarse, dim=1).numpy()
        result = feedback_rollout_statistics(transition, z0, drive, nominal, reference, weights,
            dt=dt, budget=budget, bound=task.bound, gain=gain, seed=seed,
            samples=samples, noise_var=noise_var)
        mse = torch.as_tensor(result['mse'])[None]
        return -torch.nan_to_num(mse, nan=1e12, posinf=1e12, neginf=1e12), mse

    out = optimize_icem(score, window=len(drive), coarse_factor=task.coarse, input_dim=2,
        dt=dt, budget=budget, action_max=task.bound, action_seed=action_seed,
        seed_candidates=seed_candidates, **settings)
    out['feedforward_energy'] = out.pop('energy')
    return out


class FeedbackTrackingController(TrackingController):
    """Shared MPC variant; use spike-derived feedback at each 5-ms SNN input bin."""

    def __init__(self, *args, feedback_gain, **kwargs):
        if not np.isfinite(feedback_gain) or feedback_gain < 0:
            raise ValueError('Feedback gain must be finite and nonnegative')
        super().__init__(*args, **kwargs)
        self.feedback_gain = float(feedback_gain)

    def plan_tracking(self, task, t, k, target, pool, budget_left):
        horizon = task.window-k
        drive = np.stack([ev.evidence_drive(t+j, pool) for j in range(horizon)])
        z0 = self.m.copy()
        if target == 0:
            z0, drive = z0[::-1].copy(), drive[:, ::-1].copy()
        seeds = ev.seed_pushes(horizon, budget_left, self.dt, task.coarse)
        seeds = np.concatenate([seeds, -seeds, np.zeros_like(seeds[:1])])
        if self._last_plan is not None:
            seeds = np.concatenate([seeds,
                ev.carry_over_plan(self._last_plan, k, target, task.coarse)[None]])
        weights = (np.arange(k, task.window) >= task.warmup).astype(float)
        settings = {key: ev.PROTOCOL[f'icem_{key}'] for key in (
            'num_samples', 'num_elites', 'num_iterations', 'init_std', 'alpha',
            'noise_beta', 'factor_decrease', 'frac_elites_reused')}
        out = feedback_tracking_plan(self.transition['plan'], z0, drive, task.reference()[k:],
            weights, task=task, dt=self.dt, budget=budget_left, gain=self.feedback_gain,
            seed=self.seed, action_seed=self.seed*1000+k, seed_candidates=seeds, settings=settings)
        nominal = out['u'][:, ::-1].copy() if target == 0 else out['u']
        self._last_plan = (k, nominal)
        return nominal

    def apply_input(self, planned, *, task, k, target, remaining):
        state, nominal = self.m, np.asarray(planned)
        if target == 0:
            state, nominal = state[::-1].copy(), nominal[::-1].copy()
        action = feedback_input(torch.as_tensor(nominal, dtype=torch.float64),
            torch.as_tensor(state, dtype=torch.float64), float(task.reference()[k]), remaining,
            gain=self.feedback_gain, bound=task.bound, dt=self.dt).numpy()
        return action[::-1].copy() if target == 0 else action
