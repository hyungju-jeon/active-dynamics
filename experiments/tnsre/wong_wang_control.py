"""Matched control diagnostics, independent constraints, and activity tracking.

Identification estimates are frozen. All learned methods use the same observer,
planner settings, and task. The full SNN is used only for evaluation; its latent
proxy is unavailable to controllers except explicitly labelled diagnostics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch

from actdyn.utils.validation import optimize_icem
from experiments.tnsre import eval_spiking_sessions as ev


@dataclass(frozen=True)
class ControlTask:
    """Task contract; dt uses 100-ms units, and z is the NMDA gating proxy."""

    name: str = 'overturn'
    bound: float = 1.
    budget: float = 10.
    window: int = 400
    settle: int = 100
    warmup: int = 100
    tolerance: float = .35
    replan: int = 40
    coarse: int = 10

    def __post_init__(self):
        if self.name not in ('overturn', 'neutral', 'moderate', 'switch'):
            raise ValueError(self.name)
        if self.bound <= 0 or self.budget < 0 or not 0 <= self.warmup < self.window:
            raise ValueError('Invalid constraints or scoring interval')
        if self.window % self.replan or self.replan % self.coarse:
            raise ValueError('Window, replanning, and coarse bins must divide exactly')

    @property
    def key(self):
        return f'{self.name}_a{self.bound:g}_b{self.budget:g}'

    def reference(self) -> np.ndarray:
        """Desired target-minus-opponent gap after each bin, float64 (window,)."""
        if self.name == 'neutral':
            return np.zeros(self.window)
        if self.name == 'moderate':
            return np.ones(self.window)
        if self.name == 'switch':
            return np.where(np.arange(self.window) < self.window // 2, 1., -1.)
        return np.full(self.window, 2.)


def tracking_metrics(gaps: np.ndarray, task: ControlTask) -> dict:
    """Score the full fixed interval, including excursions after target entry.

    Gaps are true target-minus-opponent NMDA proxies after each control bin.
    Success requires >=90% of scored bins within the preset tolerance. RMSE is
    the primary tracking endpoint; first entry is never an early stopping rule.
    """
    gaps = np.asarray(gaps, dtype=np.float64)
    if gaps.shape != (task.window,) or not np.isfinite(gaps).all():
        raise ValueError('Need one finite gap per control bin')
    error = gaps[task.warmup:] - task.reference()[task.warmup:]
    occupancy = float(np.mean(np.abs(error) <= task.tolerance))
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                occupancy=occupancy, success=int(occupancy >= .9),
                max_error=float(np.max(np.abs(error))))


def rollout_statistics(transition, z0, drive, actions, *, dt, seed, samples=8,
                       noise_var=.1, reference=None, weights=None, threshold=2.,
                       sharpness=4.) -> dict[str, np.ndarray]:
    """Predict candidate outcomes using common additive-noise draws.

    z0 (2,), drive (T,2), actions (K,W,2), W<=T. Float32 model rollouts.
    With reference/weights (T,), return weighted expected tracking MSE. Without
    them, return first-passage soft/hard success. Outputs have shape (K,).
    """
    w = torch.as_tensor(np.asarray(drive), dtype=torch.float32)
    u = torch.as_tensor(np.asarray(actions), dtype=torch.float32)
    z = torch.as_tensor(np.asarray(z0), dtype=torch.float32)
    if u.ndim != 3 or u.shape[-1] != 2 or w.ndim != 2 or w.shape[-1] != 2:
        raise ValueError('Need actions (K,W,2) and drive (T,2)')
    if u.shape[1] > len(w) or z.shape != (2,):
        raise ValueError('Control cannot exceed the prediction horizon')
    rng = torch.Generator(device='cpu').manual_seed(int(seed))
    noise = torch.randn(len(w), samples, 2, generator=rng) * np.sqrt(noise_var * dt)
    z = z.expand(len(u), samples, 2).clone()
    best = torch.full((len(u), samples), -float('inf'))
    mse = torch.zeros_like(best)
    if reference is not None:
        reference = torch.as_tensor(reference, dtype=torch.float32)
        weights = torch.as_tensor(weights, dtype=torch.float32)
        if reference.shape != (len(w),) or weights.shape != reference.shape or weights.sum() <= 0:
            raise ValueError('Need a reference and positive scoring mass over the horizon')
    with torch.no_grad():
        for t in range(len(w)):
            inp = w[t].expand(len(u), samples, 2)
            if t < u.shape[1]:
                inp = inp + u[:, t, None, :]
            z = transition(z.reshape(-1, 2), inp.reshape(-1, 2)).reshape(len(u), samples, 2) + noise[t]
            gap = z[..., 1] - z[..., 0]
            best = torch.maximum(best, gap)
            if reference is not None:
                mse += weights[t] * (gap - reference[t])**2
        if reference is not None:
            return {'mse': (mse.mean(-1) / weights.sum()).numpy()}
        return {'soft': torch.sigmoid(sharpness * (best - threshold)).mean(-1).numpy(),
                'hard': (best > threshold).float().mean(-1).numpy(),
                'peak_gap': best.mean(-1).numpy()}


def tracking_plan(transition, z0, drive, reference, weights, *, task: ControlTask,
                  dt: float, budget: float, seed: int, action_seed: int,
                  seed_candidates=None, settings=None):
    """Minimize expected squared tracking error with the shared iCEM optimizer.

    Uses the same action sampler and bounds as the manuscript reversal planner.
    The scorer is a distinct mathematical function, without a first-passage
    surrogate or an early stop after entering the target band.
    """
    window = len(drive)
    settings = {} if settings is None else dict(settings)
    samples = settings.pop('mc_samples', int(ev.PROTOCOL['planner_mc_samples']))
    noise_var = settings.pop('noise_var', float(ev.PROTOCOL['planner_noise_var']))

    def score(coarse):
        actions = coarse[0].repeat_interleave(task.coarse, dim=1).numpy()
        result = rollout_statistics(transition, z0, drive, actions, dt=dt, seed=seed,
            samples=samples, noise_var=noise_var, reference=reference, weights=weights)
        mse = torch.as_tensor(result['mse'])[None]
        # Divergent predictions remain poor candidates, rather than becoming NaN elites.
        score = -torch.nan_to_num(mse, nan=1e12, posinf=1e12, neginf=1e12)
        return score, mse

    out = optimize_icem(score, window=window, coarse_factor=task.coarse, input_dim=2,
        dt=dt, budget=budget, action_max=task.bound, action_seed=action_seed,
        seed_candidates=seed_candidates, **settings)
    return out


class TrackingController(ev.IcemController):
    """Frozen learned dynamics and spike-count EKF, with a tracking objective."""

    def apply_input(self, planned, *, task, k, target, remaining):
        """Original controller: hold the planned input until its next update."""
        return planned

    def plan_tracking(self, task, t, k, target, pool, budget_left):
        horizon = task.window - k
        drive = np.stack([ev.evidence_drive(t + j, pool) for j in range(horizon)])
        z0 = self.m.copy()
        if target == 0:
            z0, drive = z0[::-1].copy(), drive[:, ::-1].copy()
        seeds = ev.seed_pushes(horizon, budget_left, self.dt, task.coarse)
        # Both directions and no input are useful for regulation; all models get them.
        seeds = np.concatenate([seeds, -seeds, np.zeros_like(seeds[:1])])
        if self._last_plan is not None:
            seeds = np.concatenate([seeds,
                ev.carry_over_plan(self._last_plan, k, target, task.coarse)[None]])
        weights = (np.arange(k, task.window) >= task.warmup).astype(float)
        settings = {key: ev.PROTOCOL[f'icem_{key}'] for key in (
            'num_samples', 'num_elites', 'num_iterations', 'init_std', 'alpha',
            'noise_beta', 'factor_decrease', 'frac_elites_reused')}
        out = tracking_plan(self.transition['plan'], z0, drive, task.reference()[k:], weights,
            task=task, dt=self.dt, budget=budget_left, seed=self.seed,
            action_seed=self.seed * 1000 + k, seed_candidates=seeds, settings=settings)
        u = out['u'][:, ::-1].copy() if target == 0 else out['u']
        self._last_plan = (k, u)
        return u


class CountObserver:
    """Static Poisson readout inversion with exponential count smoothing.

    This baseline observer uses the same calibrated C,b and spike counts, and
    no learned dynamics or privileged SNN states. tau is in seconds. A bounded
    Newton solve estimates z from the smoothed rates, with a small ridge.
    """

    def __init__(self, C, b, dt, tau=.05):
        self.C, self.b, self.dt = np.asarray(C), np.asarray(b), float(dt)
        self.decay = np.exp(-.1 * dt / tau)
        self.m = np.array([-2., -2.])
        self.rate = np.exp(self.C @ self.m + self.b)

    def observe(self, counts, drive=None):
        self.rate = self.decay * self.rate + (1-self.decay) * np.asarray(counts) / self.dt
        for _ in range(4):
            rate = np.exp(np.clip(self.C @ self.m + self.b, -20., 20.))
            gradient = self.C.T @ (rate-self.rate)
            hessian = self.C.T @ (rate[:, None] * self.C) + 1e-3*np.eye(2)
            step = np.clip(np.linalg.solve(hessian, gradient), -.5, .5)
            self.m = np.clip(self.m-step, -2., 2.)


def baseline_action(name, *, task, k, target, dt, remaining, estimate=None,
                    integral=0., parameters=None):
    """Pulse, relay, and PI controls; all obey the same instantaneous/energy bounds.

    Tracking pulse direction follows the known reference sign at each bin. Relay
    and PI use the count observer's gap; neither receives the true state.
    Returns float64 input (2,) and the updated scalar integral.
    """
    p = {} if parameters is None else parameters
    desired = float(task.reference()[k])
    sign = 1. if task.name == 'overturn' or desired >= 0 else -1.
    amp = 0.
    if name == 'front':
        amp = sign * task.bound
    elif name == 'spread':
        amp = sign * min(task.bound, np.sqrt(task.budget/(dt*task.window)))
    elif name == 'pulse':
        start = int(p.get('delay_bins', 0))
        stop = start + int(p.get('duration_bins', task.window))
        amp = sign * task.bound * float(p.get('amplitude', 1.)) if start <= k < stop else 0.
    elif name in ('relay', 'pi'):
        if estimate is None:
            raise ValueError('Feedback baseline requires spike-derived state estimate')
        error = desired - float(estimate[target]-estimate[1-target])
        if name == 'relay':
            amp = task.bound * np.sign(error) if abs(error) > p.get('deadband', .15) else 0.
        else:
            # Conditional integration prevents windup at the amplitude limit.
            kp, ki = p.get('kp', .5), p.get('ki', .5)
            proposal = integral + dt*.1*error
            raw = kp*error + ki*proposal
            if abs(raw) <= task.bound or np.sign(error) != np.sign(raw):
                integral = proposal
            amp = np.clip(kp*error+ki*integral, -task.bound, task.bound)
    elif name != 'none':
        raise ValueError(name)
    amp = np.clip(amp, -min(task.bound, np.sqrt(max(remaining, 0.)/dt)),
                  min(task.bound, np.sqrt(max(remaining, 0.)/dt)))
    return amp * ev.target_direction(target), integral


def task_protocol(task):
    """Set the legacy planner's experiment-boundary globals for this explicit task."""
    ev.PROTOCOL.update(action_max=task.bound, task_checkpoints=[20])
    ev.PROTOCOL['window'] = dict(ev.PROTOCOL['window'], overturn=task.window)
    ev.PROTOCOL['settle'] = task.settle
    ev.PROTOCOL['replan_interval'] = dict(ev.PROTOCOL['replan_interval'], overturn=task.replan)
    ev.PROTOCOL['icem_coarse_factor'] = task.coarse


def run_session(net, task, seed, *, controller=None, baseline='none', parameters=None,
                trace=None):
    """One paired SNN session, starting control at the first uncontrolled decision.

    All controllers receive the decision event/target. Only evaluation reads z.
    Overturn retains the manuscript's first-passage rule. Tracking always runs
    its entire fixed window. trace records controls, observations, and states.
    """
    dt = float(net.dt_latent)
    task_protocol(task)
    pool = ev.evidence_pool(seed)
    z = net.reset(seed)
    for t in range(int(ev.PROTOCOL['max_session_bins'])):
        evidence = ev.evidence_drive(t, pool)
        counts, z = net.step(np.zeros(2), evidence=evidence)
        # Record the applied step even if the following observer update fails.
        if trace is not None:
            trace.append(dict(bin=t, u=np.zeros(2), z=z.copy(), evidence=evidence, counts=counts.copy()))
        if controller is not None:
            controller.observe(counts, evidence)
        if abs(z[0]-z[1]) > ev.PROTOCOL['decision_gap']:
            break
    else:
        return dict(valid=0, success=0, energy=0., onset=-1, target=-1,
                    rmse=None, occupancy=None, outcome_bin=-1)
    onset = t+1
    target = 1 if z[0] > z[1] else 0
    energy, integral, plan, plan_k, success, outcome = 0., 0., None, 0, 0, -1
    gaps = []
    last = task.window + (task.settle if task.name == 'overturn' else 0)
    for k in range(last):
        u = np.zeros(2)
        if k < task.window and task.budget-energy > 1e-9:
            if isinstance(controller, ev.IcemController):
                if k % task.replan == 0:
                    if task.name == 'overturn':
                        plan = controller.plan('overturn', onset+k, k, target, pool, task.budget-energy)
                    else:
                        plan = controller.plan_tracking(task, onset+k, k, target, pool, task.budget-energy)
                    plan_k = k
                u = plan[k-plan_k]
                if isinstance(controller, TrackingController):
                    u = controller.apply_input(u, task=task, k=k, target=target,
                                               remaining=task.budget-energy)
            else:
                u, integral = baseline_action(baseline, task=task, k=k, target=target,
                    dt=dt, remaining=task.budget-energy, integral=integral,
                    estimate=None if controller is None else controller.m, parameters=parameters)
            if not np.isfinite(u).all() or np.linalg.norm(u) > task.bound+1e-6:
                raise RuntimeError(f'Invalid input: {u}')
            cost = dt*float(u@u)
            if cost > task.budget-energy:
                u *= np.sqrt(max(task.budget-energy, 0.) / max(cost, 1e-12))
            energy += dt*float(u@u)
        evidence = ev.evidence_drive(onset+k, pool)
        counts, z = net.step(u, evidence=evidence)
        if trace is not None:
            trace.append(dict(bin=onset+k, u=u.copy(), z=z.copy(), evidence=evidence, counts=counts.copy()))
        if controller is not None:
            controller.observe(counts, u+evidence)
        gap = float(z[target]-z[1-target])
        gaps.append(gap)
        if task.name == 'overturn' and gap > ev.PROTOCOL['decision_gap']:
            success, outcome = 1, onset+k+1
            break
    result = dict(valid=1, success=success, energy=energy, onset=onset, target=target,
                  rmse=None, occupancy=None, outcome_bin=outcome)
    if task.name != 'overturn':
        result.update(tracking_metrics(np.array(gaps), task))
    return result
