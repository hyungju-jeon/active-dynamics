"""Identification over decision sessions, then a decision task, on the Wang (2002) network.

Identification (``experiments.tnsre.exp_spiking_sessions``): every agent runs 20
decision sessions. A session starts in the quiet state; a decision is made when
``|z_1 - z_2| > 2``; 200 ms later, or after 2 s without a decision, the circuit
resets. The agents know this rule. Every identified model is the learner's
reduced Wong-Wang drift with the input inside the gain (``wong_wang_inside_gain``)
and the estimate a run held after ``k`` completed sessions.

This module scores those models against the spiking network:

1. **Rollout R2** on held-out network trajectories under random piecewise-constant
   pool drives (stage ``test_data`` records them) after every identification
   session (stage ``r2``).

2. **Task sessions** (stage ``task``). After ``k`` identification sessions the
   agent's parameters are frozen and it runs task sessions on the network. In
   each, 6 pA of evidence drives a random pool for 1 s. Two tasks:

   * ``force``: make the other pool win the first decision. The agent controls
     from evidence onset for 1 s; success means the first decision (within the
     2 s session) is the target pool.
   * ``overturn``: the uncontrolled circuit decides first; at the decision the
     agent controls for 2 s to make the other pool win. Success means the target
     pool leads by more than 2 within 2.5 s of the decision.

   The input energy ``dt sum_t |u_t|^2`` of a session is at most the budget.
   The agent is closed loop: its state filter (the agents' EKF with the frozen
   estimate and the run's calibrated readout,
   :func:`actdyn.utils.validation.poisson_ekf_update`) tracks the spike counts,
   and every 100 ms (force) or 200 ms (overturn) it replans the rest of the window
   with iCEM (:func:`actdyn.utils.validation.icem_reversal_plan`) from the filtered
   state with the remaining budget. The search is seeded with simple pushes toward
   the target (:func:`seed_pushes`) and, for the ``warm`` controller, with the rest
   of the previous plan. The planner maximizes the model-predicted probability of
   the task's own success rule (first decision for force; a lead above 2 at any
   time for overturn) over the rest of the task session. Evidence is known to the
   planner. The drift is symmetric under pool exchange, so a plan for pool 1 is
   the pool-2 plan in swapped coordinates.

   References on the same sessions: no input, an even push toward the target for
   the whole window, a full-amplitude push from onset until the budget is spent,
   and the iCEM controller with the reduced model fitted to network data
   (``reduced_fit``, with each seed's readout).

Task sessions of a given seed are identical for every controller until the
control starts (paired design): network noise is fixed by the seed.

Stages (``--stage``):
    test_data  held-out trajectories                         (one network)
    r2         R2 after every identification session         (no network)
    task       closed-loop task sessions                     (sharded, one network per worker)
    score      success summaries                             (no network)
    examples   a few task sessions recorded bin by bin       (one network; for figures)

Example:
    E=results/tnsre/20260924_snn_sessions_m2/eval; RUNS=results/tnsre/20260924_snn_sessions_m2/tracks/wong_wang_snn_sessions_m2
    python -m experiments.tnsre.eval_spiking_sessions --stage r2 --out $E --runs-root $RUNS
    python -m experiments.tnsre.eval_spiking_sessions --stage task --out $E --runs-root $RUNS --worker-index 0 --n-workers 1
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

PRESET_ID = "tbme_wong_wang_snn_sessions_m2"
TASKS = ("force", "overturn")

PROTOCOL: dict[str, Any] = {
    # Held-out trajectories: 12 network runs under random piecewise-constant pool drives.
    "test_trajectories": "results/tnsre/20260924_snn_sessions_m2/data/test_trajectories.npz",
    "test_seeds": list(range(70000, 70012)),
    "test_bins": 400,
    "test_hold": 40,
    "test_levels": [-1.0, -0.5, 0.0, 0.5, 1.0],
    # Reduced model fitted to network data (the reference of the evaluation).
    "reduced_fit": "results/tnsre/20260924_snn_sessions_m2/reference/m2_fit.json",
    "r2_horizon": 100,
    "r2_stride": 20,
    "task_checkpoints": [0, 1, 2, 5, 20],       # completed identification sessions
    "task_sessions": 8,                        # task sessions per task
    "task_seeds": {"force": 90000, "overturn": 91000},
    "evidence_amplitude": 0.3,                 # 6 pA in action units (x 20 pA)
    "evidence_bins": 200,
    "decision_gap": 2.0,
    "max_session_bins": 400,
    # Control window: force 1 s from evidence onset; overturn 2 s from the decision
    # (with |u| <= 1 a 1 s window holds at most energy 10, and even that overturns
    # under half of the decisions, so the budget would not bind).
    "window": {"force": 200, "overturn": 400},
    "settle": 100,                             # overturn: success read until 2.5 s after the decision
    "replan_interval": {"force": 20, "overturn": 40},   # 100 ms and 200 ms
    # Budgets from a model-free calibration under these rules on separate sessions
    # (seeds 82000+ and 83000+, 8 each). Force, even push / full push from onset:
    # 0.12 / 0.00 at 0.5, 0.75 / 0.00 at 1, 1.00 / 0.12 at 2. Overturn with the 2 s
    # window: 0.00 / 0.00 at 8, - / 0.38 at 10, 0.00 / 0.88 at 12, 0.88 / 1.00 at 16.
    "budgets": {"force": [0.5, 1.0], "overturn": [10.0, 12.0]},
    "reset_state": [-2.0, -2.0],
    "reset_variance": 0.01,
    "process_var": float(np.log1p(0.1)),       # softplus(log state_noise) of the learner
    # The planner scores the task's own success rule over the rest of the task
    # session (force: to the 2 s session end; overturn: to 2.5 s after the decision).
    "planner_objective": {"force": "first_decision", "overturn": "reach"},
    "planner_noise_var": 0.1,
    "planner_mc_samples": 8,
    "icem_coarse_factor": 10,
    "icem_num_samples": 64,
    "icem_num_elites": 8,
    "icem_num_iterations": 8,
    "icem_init_std": 0.5,
    "icem_alpha": 0.1,
    "icem_noise_beta": 1.0,
    "icem_factor_decrease": 1.25,
    "icem_frac_elites_reused": 0.3,
    "icem_sharpness": 4.0,
}
MODEL_FREE = ("none", "spread", "front")
# Example task sessions recorded bin by bin for figures (stage ``examples``): one training seed's
# PALDI estimate after 20 sessions, the fit, and the best model-free push, on one session per task.
EXAMPLES: dict[str, Any] = {
    "training_seed": 0,
    "policy": "adaptive",
    "checkpoint": 20,
    "sessions": {"overturn": [91000, 12.0], "force": [90000, 1.0]},
    "model_free": {"overturn": "front", "force": "spread"},
}


# ------------------------------------------------------------------ network and held-out data
def build_network(seed: int = 0):
    """Wang (2002) network with the configuration of the experiment's preset."""
    from actdyn.environment.spiking_decision import TAU_NMDA_MS, WangDecisionNetwork
    from experiments.experiment_definitions import get_environment_preset
    from experiments.tnsre.run_tbme_experiments import configure_tbme_catalogs

    configure_tbme_catalogs(suite_entries={})
    p = get_environment_preset(PRESET_ID)
    return WangDecisionNetwork(
        n_e=int(p.spiking_n_e), n_i=int(p.spiking_n_i), f_sel=float(p.spiking_f_sel),
        w_plus=float(p.spiking_w_plus), sim_dt_ms=float(p.spiking_sim_dt_ms),
        control_bin_ms=float(p.dt) * TAU_NMDA_MS, stim_gain_pa=float(p.spiking_stim_gain_pa),
        observed_per_pool=int(p.spiking_observed_per_pool),
        background_rate_hz=float(p.spiking_background_rate_hz), seed=int(seed),
        codegen_target=str(p.spiking_codegen_target),
    )


def test_input_program(seed: int) -> np.ndarray:
    """Held-out piecewise-constant pool drives, shape (test_bins, 2)."""
    rng = np.random.default_rng(int(seed))
    n_blocks = int(np.ceil(PROTOCOL["test_bins"] / PROTOCOL["test_hold"]))
    levels = rng.choice(np.asarray(PROTOCOL["test_levels"]), size=(n_blocks, 2))
    return np.repeat(levels, int(PROTOCOL["test_hold"]), axis=0)[: int(PROTOCOL["test_bins"])]


def stage_test_data(out: Path) -> None:
    """Record the held-out trajectories: states (K, T+1, 2) and drives (K, T, 2)."""
    net = build_network()
    states, inputs = [], []
    for seed in PROTOCOL["test_seeds"]:
        program = test_input_program(seed)
        z = [net.reset(int(seed))]
        for t in range(program.shape[0]):
            z.append(net.step(program[t])[1])
        states.append(np.asarray(z))
        inputs.append(program)
    net.close()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "test_trajectories.npz", states=np.stack(states), inputs=np.stack(inputs))
    print(f"{len(states)} held-out trajectories written to {out / 'test_trajectories.npz'}")


# ------------------------------------------------------------------ runs and models
def reference_fit() -> np.ndarray:
    """Learner parameters (w_+, w_-, h_raw, gamma_raw, g_raw) of the model fitted to network data."""
    record = json.loads(Path(PROTOCOL["reduced_fit"]).read_text())
    return np.asarray(record["learner_parameters"]["values"], dtype=np.float64)


def learner_transition(dynamics_type: str, theta: np.ndarray, full_params: np.ndarray, min_embedding_dim: int,
                       dt: float, dtype: torch.dtype):
    """Learner transition ``z + dt f(z, u; theta)`` for batched or single states (float32 inside).

    Same arithmetic as :func:`actdyn.environment.vectorfield.drift_torch`; the vector
    field is built once, since the planner calls the transition in its inner loop.
    """
    from actdyn.environment.vectorfield import build_vectorfield, pad_embedding_to_params

    params = pad_embedding_to_params(torch.as_tensor(np.asarray(theta), dtype=torch.float32).reshape(1, -1),
                                     full_params=np.asarray(full_params, dtype=np.float32),
                                     min_embedding_dim=int(min_embedding_dim))
    vf = build_vectorfield(dynamics_type, params, dynamics_alpha=1.0)

    def transition(z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        d = int(z.shape[-1])
        zf = z.float().reshape(-1, d)
        uf = torch.as_tensor(u, dtype=torch.float32).expand_as(z).reshape(-1, d)
        f = vf.compute_with_input(zf, uf).reshape(z.shape)
        return z + float(dt) * f.to(dtype)

    return transition


def load_session_runs(runs_root: Path) -> list[dict[str, Any]]:
    """Estimates after each completed session, readout, and prior of every completed run."""
    runs = []
    for meta_path in sorted(runs_root.glob("*/seed_*/repeat_*/run_metadata.json")):
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "completed":
            continue
        with open(meta_path.parent / "embedding_estimate_trace.csv") as fh:
            rows = list(csv.DictReader(fh))
        est = {0: np.asarray(meta["initial_parameter"], dtype=np.float64)}
        n_e = int(est[0].shape[0])
        bins = {0: 0}
        for r in rows:
            if r["session_end"] == "True":
                k = int(r["session_index"]) + 1
                est[k] = np.array([float(r[f"e{i}"]) for i in range(n_e)])
                bins[k] = int(float(r["step"]))
        runs.append({
            "policy": str(meta["policy_id"]), "seed": int(meta["seed"]), "est": est, "bins": bins,
            "dynamics_type": str(meta["estimator_dynamics_type"]),
            "full_params": np.asarray(meta["estimator_true_params_full"], dtype=np.float64),
            "min_embedding_dim": int(meta["min_embedding_dim"]),
            "C": np.asarray(meta["observation_loading_matrix"], dtype=np.float64),
            "b": np.asarray(meta["observation_loading_bias"], dtype=np.float64),
            "session_log": meta.get("session_log", []),
        })
    return runs


# ------------------------------------------------------------------ task sessions
def evidence_pool(seed: int) -> int:
    """Pool (0 or 1) that receives the evidence in task session ``seed``."""
    return int(np.random.default_rng(int(seed)).integers(2))


def evidence_drive(t: int, pool: int) -> np.ndarray:
    u = np.zeros(2, dtype=np.float64)
    if t < int(PROTOCOL["evidence_bins"]):
        u[int(pool)] = float(PROTOCOL["evidence_amplitude"])
    return u


def target_direction(target: int) -> np.ndarray:
    """Unit input that excites the target pool and suppresses the other."""
    d = -np.ones(2) / np.sqrt(2.0)
    d[int(target)] = 1.0 / np.sqrt(2.0)
    return d


def model_free_input(name: str, task: str, k: int, target: int, budget: float, dt: float) -> np.ndarray:
    """Input of a model-free controller at bin ``k`` of the control window."""
    window = int(PROTOCOL["window"][task])
    if name == "none" or k >= window:
        return np.zeros(2)
    if name == "spread":
        return min(1.0, float(np.sqrt(budget / (dt * window)))) * target_direction(target)
    if name == "front":
        return target_direction(target) if k < int(budget / dt + 1e-9) else np.zeros(2)
    raise ValueError(f"unknown model-free controller {name!r}")


def seed_pushes(window_left: int, budget_left: float, dt: float, factor: int) -> np.ndarray:
    """Simple pushes toward pool 2 that seed the iCEM search, shape (6, window_left / factor, 2).

    Three directions (both pools, excite pool 2 only, suppress pool 1 only) times
    two shapes: full amplitude from now until the budget is spent, and an even
    push over the rest of the window. Held values are coarse (``factor`` bins).
    """
    hc = int(window_left) // int(factor)
    coarse_dt = float(dt) * int(factor)
    directions = (np.array([-1.0, 1.0]) / np.sqrt(2.0), np.array([0.0, 1.0]), np.array([-1.0, 0.0]))
    n_full = min(hc, int(np.floor(float(budget_left) / coarse_dt + 1e-9)))
    front = np.zeros(hc)
    front[:n_full] = 1.0
    if n_full < hc:
        front[n_full] = np.sqrt(max(float(budget_left) - n_full * coarse_dt, 0.0) / coarse_dt)
    even = np.full(hc, min(1.0, float(np.sqrt(float(budget_left) / (coarse_dt * hc)))))
    return np.stack([shape[:, None] * d[None, :] for d in directions for shape in (front, even)])


def carry_over_plan(last_plan: tuple[int, np.ndarray], k: int, target: int, factor: int) -> np.ndarray:
    """Rest of the previous plan from window bin ``k``, as coarse held values in planning coordinates.

    ``last_plan`` is (window bin where that plan started, its fine inputs in network
    coordinates). Planning coordinates put the target in pool 2, so a pool-1 target
    swaps the pools.
    """
    k_prev, u_prev = last_plan
    rest = u_prev[int(k) - int(k_prev) :: int(factor)]
    return rest[:, ::-1].copy() if int(target) == 0 else rest


class IcemController:
    """Closed-loop iCEM with a frozen learner model and the agents' state filter."""

    def __init__(self, theta: np.ndarray, C: np.ndarray, b: np.ndarray, dt: float, seed: int, *,
                 dynamics_type: str, full_params: np.ndarray, min_embedding_dim: int, warm_start: bool = False,
                 filter_theta: np.ndarray | None = None) -> None:
        self.theta, self.C, self.b, self.dt, self.seed = np.asarray(theta, float), C, b, float(dt), int(seed)
        # The state filter may use other parameters than the planner (diagnostics only).
        self.filter_theta = self.theta if filter_theta is None else np.asarray(filter_theta, float)
        self.m = np.asarray(PROTOCOL["reset_state"], dtype=np.float64)
        self.P = float(PROTOCOL["reset_variance"]) * np.eye(2)
        # "warm": each replan is also seeded with the rest of the previous plan (as MpcICem
        # shifts its plan); "fresh" (default): seeded with simple pushes only.
        self.warm_start = bool(warm_start)
        self._last_plan: tuple[int, np.ndarray] | None = None
        self.transition = {
            "plan": learner_transition(dynamics_type, self.theta, full_params, min_embedding_dim, self.dt,
                                       torch.float32),
            "filter": learner_transition(dynamics_type, self.filter_theta, full_params, min_embedding_dim,
                                         self.dt, torch.float64),
        }

    def observe(self, counts: np.ndarray, drive: np.ndarray) -> None:
        from actdyn.utils.validation import poisson_ekf_update

        self.m, self.P = poisson_ekf_update(self.m, self.P, counts, drive, transition=self.transition["filter"],
                                            readout_weight=self.C, readout_bias=self.b, dt=self.dt,
                                            process_var=float(PROTOCOL["process_var"]))

    def plan(self, task: str, t: int, k: int, target: int, pool: int, budget_left: float) -> np.ndarray:
        """Inputs for bins ``k..window-1`` of the control window, starting at session bin ``t``."""
        from actdyn.utils.validation import icem_reversal_plan

        window_left = int(PROTOCOL["window"][task]) - int(k)
        if task == "force":
            horizon = int(PROTOCOL["max_session_bins"]) - int(t)
        else:
            horizon = window_left + int(PROTOCOL["settle"])
        drive = np.stack([evidence_drive(t + j, pool) for j in range(horizon)])
        z0 = self.m.copy()
        if target == 0:  # plan in swapped coordinates, where the target is pool 2
            z0, drive = z0[::-1].copy(), drive[:, ::-1].copy()
        factor = int(PROTOCOL["icem_coarse_factor"])
        seeds = seed_pushes(window_left, budget_left, self.dt, factor)
        if self.warm_start and self._last_plan is not None:
            seeds = np.concatenate([seeds, carry_over_plan(self._last_plan, k, target, factor)[None]], axis=0)
        out = icem_reversal_plan(
            self.transition["plan"], z0, drive, dt=self.dt, window=window_left, coarse_factor=factor,
            budget=float(budget_left), action_max=1.0, noise_var=float(PROTOCOL["planner_noise_var"]),
            objective=str(PROTOCOL["planner_objective"][task]), success_gap=float(PROTOCOL["decision_gap"]),
            mc_samples=int(PROTOCOL["planner_mc_samples"]), mc_seed=self.seed, action_seed=self.seed * 1000 + int(k),
            num_samples=int(PROTOCOL["icem_num_samples"]), num_elites=int(PROTOCOL["icem_num_elites"]),
            num_iterations=int(PROTOCOL["icem_num_iterations"]), init_std=float(PROTOCOL["icem_init_std"]),
            alpha=float(PROTOCOL["icem_alpha"]), noise_beta=float(PROTOCOL["icem_noise_beta"]),
            factor_decrease=float(PROTOCOL["icem_factor_decrease"]),
            frac_elites_reused=float(PROTOCOL["icem_frac_elites_reused"]),
            sharpness=float(PROTOCOL["icem_sharpness"]), seed_candidates=seeds,
        )
        u = out["u"]
        u = u[:, ::-1].copy() if target == 0 else u
        self._last_plan = (int(k), u)
        return u


def run_task_session(net, task: str, seed: int, budget: float, controller: str,
                     model: IcemController | None, trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """One task session on the network; ``model`` is None for model-free controllers.

    With ``trace`` (a list), every bin appends ``{"bin", "z", "evidence", "u"}``: the latent after
    the bin, the evidence drive, and the controller input (zero before the control window).
    """
    dt = float(net.dt_latent)
    gap_thr = float(PROTOCOL["decision_gap"])
    pool = evidence_pool(seed)
    z = net.reset(int(seed))
    t, onset = 0, 0
    if task == "overturn":
        while t < int(PROTOCOL["max_session_bins"]):
            drive = evidence_drive(t, pool)
            counts, z = net.step(drive)
            if model is not None:
                model.observe(counts, drive)
            if trace is not None:
                trace.append({"bin": t, "z": np.asarray(z, dtype=np.float64), "evidence": drive, "u": np.zeros(2)})
            t += 1
            if abs(float(z[0] - z[1])) > gap_thr:
                break
        else:
            return {"valid": 0, "evidence_pool": pool, "target": -1, "onset": -1, "success": 0,
                    "energy": 0.0, "outcome_bin": -1, "first_decision": 0}
        onset = t
        target = 1 if float(z[0] - z[1]) > 0 else 0  # overturn the decision that was made
        last = onset + int(PROTOCOL["window"][task]) + int(PROTOCOL["settle"])
    else:
        target = 1 - pool
        last = int(PROTOCOL["max_session_bins"])
    energy, plan, plan_k = 0.0, None, 0
    first_decision, success, outcome_bin = 0, 0, -1
    window, replan = int(PROTOCOL["window"][task]), int(PROTOCOL["replan_interval"][task])
    for t in range(onset, last):
        k = t - onset
        u = np.zeros(2)
        if k < window:
            if model is None:
                u = model_free_input(controller, task, k, target, budget, dt)
            else:
                if k % replan == 0 and budget - energy > 1e-9:
                    plan, plan_k = model.plan(task, t, k, target, pool, budget - energy), k
                elif budget - energy <= 1e-9:
                    plan = None
                if plan is not None:
                    u = plan[k - plan_k]
            step_energy = dt * float(u @ u)
            if energy + step_energy > budget:  # numerical guard; plans are budget-projected
                u = u * np.sqrt(max(budget - energy, 0.0) / max(step_energy, 1e-12))
                step_energy = dt * float(u @ u)
            energy += step_energy
        drive = evidence_drive(t, pool) + u
        counts, z = net.step(drive)
        if model is not None:
            model.observe(counts, drive)
        if trace is not None:
            trace.append({"bin": t, "z": np.asarray(z, dtype=np.float64), "evidence": evidence_drive(t, pool),
                          "u": np.asarray(u, dtype=np.float64)})
        lead = float(z[target] - z[1 - target])
        if task == "force":
            if abs(lead) > gap_thr:
                first_decision = 1 if lead > 0 else -1
                success, outcome_bin = int(lead > 0), t + 1
                break
        elif lead > gap_thr:
            success, outcome_bin = 1, t + 1
            break
    return {"valid": 1, "evidence_pool": pool, "target": target, "onset": onset, "success": success,
            "energy": energy, "outcome_bin": outcome_bin, "first_decision": first_decision}


# ------------------------------------------------------------------ jobs
def controller_units(runs: list[dict[str, Any]], theta_fit: np.ndarray | None = None) -> list[dict[str, Any]]:
    """Every controller that runs task sessions: learned models, per-seed prior and fit, model-free.

    ``theta_fit`` overrides the reference fit (:func:`reference_fit`).
    """
    units = []
    seeds_done = set()
    fit = reference_fit() if theta_fit is None else np.asarray(theta_fit, float)
    for run in runs:
        dyn = {"dynamics_type": run["dynamics_type"], "full_params": run["full_params"],
               "min_embedding_dim": run["min_embedding_dim"]}
        for k in PROTOCOL["task_checkpoints"]:
            if k == 0 or k not in run["est"]:
                continue
            units.append({"controller": "icem", "policy": run["policy"], "seed": run["seed"], "checkpoint": k,
                          "theta": run["est"][k], "C": run["C"], "b": run["b"], **dyn})
        if run["seed"] not in seeds_done:  # the prior and the fit depend on the seed only
            seeds_done.add(run["seed"])
            units.append({"controller": "icem", "policy": "prior", "seed": run["seed"], "checkpoint": 0,
                          "theta": run["est"][0], "C": run["C"], "b": run["b"], **dyn})
            units.append({"controller": "icem", "policy": "reduced_fit", "seed": run["seed"], "checkpoint": -1,
                          "theta": fit, "C": run["C"], "b": run["b"], **dyn})
    for name in MODEL_FREE:
        units.append({"controller": name, "policy": name, "seed": -1, "checkpoint": -1, "theta": None})
    return units


def task_jobs(units: list[dict[str, Any]]) -> list[tuple[int, str, float, int]]:
    """(unit index, task, budget, task-session seed) for every task session."""
    jobs = []
    for u_idx, unit in enumerate(units):
        for task in TASKS:
            for budget in PROTOCOL["budgets"][task]:
                if unit["controller"] == "none" and budget != PROTOCOL["budgets"][task][0]:
                    continue  # no input does not depend on the budget
                for j in range(int(PROTOCOL["task_sessions"])):
                    jobs.append((u_idx, task, float(budget), int(PROTOCOL["task_seeds"][task]) + j))
    return jobs


def _job_key(unit: dict[str, Any], task: str, budget: float, seed: int) -> tuple:
    return (unit["controller"], unit["policy"], int(unit["seed"]), int(unit["checkpoint"]), task, float(budget), int(seed))


TASK_FIELDS = ["controller", "policy", "seed", "checkpoint", "task", "budget", "task_seed", "valid", "evidence_pool",
               "target", "onset", "success", "energy", "outcome_bin", "first_decision", "w_plus", "w_minus", "theta"]
CONTROLLERS = ("fresh", "warm")


def controller_paths(out: Path, controller: str) -> dict[str, Path]:
    """Task-session and summary files of a controller variant ("fresh" keeps the original names)."""
    if controller == "fresh":
        return {"task": out / "task", "summary": out / "task_summary.csv"}
    return {"task": out / f"task_{controller}", "summary": out / f"task_summary_{controller}.csv"}


def stage_task(out: Path, runs_root: Path, worker_index: int, n_workers: int, controller: str = "fresh",
               checkpoints: Sequence[int] | None = None) -> None:
    """Task sessions of every unit (or of units at ``checkpoints``: 0 = prior, -1 = fit and model-free)."""
    runs = load_session_runs(runs_root)
    units = controller_units(runs)
    jobs = task_jobs(units)
    if controller != "fresh":  # model-free controllers do not plan; the fresh run holds them
        jobs = [job for job in jobs if units[job[0]]["theta"] is not None]
    if checkpoints is not None:
        keep = {int(k) for k in checkpoints}
        jobs = [job for job in jobs if int(units[job[0]]["checkpoint"]) in keep]
    jobs = jobs[int(worker_index) :: int(n_workers)]
    path = controller_paths(out, controller)["task"] / f"worker_{int(worker_index):03d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()  # sessions already written by any worker (resumes survive a change of shards)
    for other in sorted(path.parent.glob("worker_*.csv")):
        with open(other) as fh:
            done |= {(r["controller"], r["policy"], int(r["seed"]), int(r["checkpoint"]), r["task"], float(r["budget"]),
                      int(r["task_seed"])) for r in csv.DictReader(fh)}
    net = build_network()
    dt = float(net.dt_latent)
    new_file = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TASK_FIELDS)
        if new_file:
            writer.writeheader()
        for u_idx, task, budget, seed in jobs:
            unit = units[u_idx]
            if _job_key(unit, task, budget, seed) in done:
                continue
            model = None
            if unit["theta"] is not None:
                model = IcemController(unit["theta"], unit["C"], unit["b"], dt, seed=seed,
                                       dynamics_type=unit["dynamics_type"], full_params=unit["full_params"],
                                       min_embedding_dim=unit["min_embedding_dim"], warm_start=(controller == "warm"))
            res = run_task_session(net, task, seed, budget, unit["controller"], model)
            theta = unit["theta"] if unit["theta"] is not None else [np.nan, np.nan]
            writer.writerow({"controller": unit["controller"], "policy": unit["policy"], "seed": unit["seed"],
                             "checkpoint": unit["checkpoint"], "task": task, "budget": budget, "task_seed": seed,
                             **res, "w_plus": float(theta[0]), "w_minus": float(theta[1]),
                             "theta": " ".join(f"{x:.6g}" for x in np.ravel(theta))})
            fh.flush()
    net.close()
    print(f"worker {worker_index}: {len(jobs)} task sessions written to {path}")


def stage_examples(out: Path, runs_root: Path) -> None:
    """Record the example task sessions of ``EXAMPLES`` bin by bin (warm controller for the models)."""
    runs = {(r["policy"], r["seed"]): r for r in load_session_runs(runs_root)}
    run = runs[(EXAMPLES["policy"], int(EXAMPLES["training_seed"]))]
    dyn = {"dynamics_type": run["dynamics_type"], "full_params": run["full_params"],
           "min_embedding_dim": run["min_embedding_dim"]}
    controllers = {"learned": run["est"][int(EXAMPLES["checkpoint"])], "fit": reference_fit()}
    net = build_network()
    dt = float(net.dt_latent)
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for task, (seed, budget) in EXAMPLES["sessions"].items():
        for name in ("learned", "fit", "none", EXAMPLES["model_free"][task]):
            model = None
            if name in controllers:
                model = IcemController(controllers[name], run["C"], run["b"], dt, seed=int(seed), warm_start=True, **dyn)
            trace: list[dict[str, Any]] = []
            res = run_task_session(net, task, int(seed), float(budget), "icem" if model else name, model, trace)
            key = f"{task}_{name}"
            arrays[f"{key}_z"] = np.stack([r["z"] for r in trace])
            arrays[f"{key}_evidence"] = np.stack([r["evidence"] for r in trace])
            arrays[f"{key}_u"] = np.stack([r["u"] for r in trace])
            rows.append({"task": task, "controller": name, "task_seed": seed, "budget": budget, **res})
    net.close()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "example_sessions.npz", **arrays)
    _write_csv(out / "example_sessions.csv", rows)
    print(f"{len(rows)} example sessions written to {out}")


# ------------------------------------------------------------------ R2 and scores
def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stage_r2(out: Path, runs_root: Path) -> None:
    from actdyn.utils.validation import rollout_r2_on_trajectories

    test = dict(np.load(PROTOCOL["test_trajectories"]))
    runs = load_session_runs(runs_root)
    if not runs:
        raise FileNotFoundError(f"No completed runs under {runs_root}")
    labels = [(run, k) for run in runs for k in sorted(run["est"])]
    est = np.array([run["est"][k] for run, k in labels])
    r2 = rollout_r2_on_trajectories(
        torch.as_tensor(est, dtype=torch.float32), dynamics_type=runs[0]["dynamics_type"],
        full_params=runs[0]["full_params"], min_embedding_dim=runs[0]["min_embedding_dim"],
        states=test["states"], inputs=test["inputs"], dt=0.05,
        horizon=int(PROTOCOL["r2_horizon"]), stride=int(PROTOCOL["r2_stride"]),
    )
    rows = [{"policy": run["policy"], "seed": run["seed"], "sessions": k, "bins": run["bins"][k],
             "w_plus": float(e[0]), "w_minus": float(e[1]), "r2": float(v), "theta": " ".join(f"{x:.6g}" for x in e)}
            for (run, k), e, v in zip(labels, est, r2)]
    _write_csv(out / "r2_per_session.csv", rows)
    session_rows = [{"policy": run["policy"], "seed": run["seed"], **s} for run in runs for s in run["session_log"]]
    _write_csv(out / "identification_sessions.csv", session_rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps({"protocol": PROTOCOL, "preset": PRESET_ID}, indent=1))
    print(f"R2 of {len(rows)} estimates from {len(runs)} runs written to {out}")


def stage_score(out: Path, controller: str = "fresh") -> None:
    paths = controller_paths(out, controller)
    rows = []
    for path in sorted(paths["task"].glob("worker_*.csv")):
        with open(path) as fh:
            rows.extend(csv.DictReader(fh))
    if controller != "fresh":  # model-free references from the fresh run (they do not plan)
        for path in sorted((out / "task").glob("worker_*.csv")):
            with open(path) as fh:
                rows.extend(r for r in csv.DictReader(fh) if r["controller"] in MODEL_FREE)
    valid = [r for r in rows if r["valid"] == "1"]
    groups: dict[tuple, list[dict[str, str]]] = {}
    for r in valid:
        groups.setdefault((r["policy"], int(r["checkpoint"]), r["task"], float(r["budget"])), []).append(r)
    summary = []
    for (policy, ck, task, budget), rs in sorted(groups.items()):
        per_seed: dict[int, list[int]] = {}
        for r in rs:
            per_seed.setdefault(int(r["seed"]), []).append(int(r["success"]))
        seed_means = np.array([np.mean(v) for v in per_seed.values()])
        summary.append({"policy": policy, "checkpoint": ck, "task": task, "budget": budget,
                        "success": float(np.mean([int(r["success"]) for r in rs])),
                        "sem_over_seeds": float(seed_means.std(ddof=1) / np.sqrt(seed_means.size)) if seed_means.size > 1 else 0.0,
                        "n_seeds": int(seed_means.size), "n_sessions": len(rs),
                        "energy_mean": float(np.mean([float(r["energy"]) for r in rs]))})
    _write_csv(paths["summary"], summary)
    print(f"{len(valid)} valid task sessions of {len(rows)}; summary in {paths['summary']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=["test_data", "r2", "task", "score", "examples"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--controller", default="fresh", choices=CONTROLLERS)
    parser.add_argument("--checkpoints", default=None,
                        help="comma-separated unit checkpoints to run (0 = prior, -1 = fit); default all")
    args = parser.parse_args(argv)
    if args.stage in ("r2", "task", "examples") and args.runs_root is None:
        parser.error("--runs-root is required for this stage")
    if args.stage == "test_data":
        stage_test_data(args.out)
    elif args.stage == "r2":
        stage_r2(args.out, args.runs_root)
    elif args.stage == "examples":
        stage_examples(args.out, args.runs_root)
    elif args.stage == "task":
        checkpoints = None if args.checkpoints is None else [int(k) for k in args.checkpoints.split(",")]
        stage_task(args.out, args.runs_root, args.worker_index, args.n_workers, args.controller, checkpoints)
    else:
        stage_score(args.out, args.controller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
