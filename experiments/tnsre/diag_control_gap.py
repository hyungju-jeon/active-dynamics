"""Why control with learned M2 models stays below control with the M2 fit.

After 20 sessions PALDI's M2 estimates reach the fit's held-out R2 but overturn
far fewer decisions (``eval_spiking_sessions``). This diagnostic runs the same
overturn task sessions (same fresh controller, sessions, and budgets) with
models that mix a run's estimate and the reference fit:

* ``learned+fit:<p>``: the learned estimate with parameter ``p`` set to the fit's value;
* ``fit+learned:<p>``: the fit with parameter ``p`` set to the learned value;
* ``plan:learned/filter:fit`` and ``plan:fit/filter:learned``: the planner and the
  state filter use different models.

``learned`` and ``fit`` themselves are the ``adaptive``/20 and ``reduced_fit`` rows
of the main M2 evaluation (identical sessions and noise). Parameters are the
learner's raw coordinates (w_+, w_-, h_raw, gamma_raw, g_raw).

A second check asks whether the models predict the network in the regime the
overturn task uses: ``record`` runs overturn trials on separate seeds (the
uncontrolled decision, then a full push of energy 12 toward the other pool in
one of three directions, then free evolution) and ``regime_r2`` scores the
learned models and the fit on them with the held-out rollout R2.

Stages: ``task`` (sharded, one network per worker), ``score``, ``record``, ``regime_r2``.

Example:
    python -m experiments.tnsre.diag_control_gap --stage task --out results/tnsre/20260924_snn_sessions_m2/diagnostics/control_gap \\
        --worker-index 0 --n-workers 1
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

DIAG: dict[str, Any] = {
    "runs_root": "results/tnsre/20260924_snn_sessions_m2/tracks/wong_wang_snn_sessions_m2",
    "policy": "adaptive",
    "checkpoint": 20,
    "task": "overturn",
    "budgets": [12.0],  # the budget with the largest gap (0.66 vs 0.95); a first run also had 10
    "parameters": ["w_plus", "w_minus", "h_raw", "gamma_raw", "g_raw"],
    # Control-regime recordings: seeds separate from the task sessions.
    "regime_seeds": list(range(84000, 84008)),
    "regime_budget": 12.0,
    "regime_bins": 500,  # recorded after the decision
    "regime_directions": {"both": [-0.70710678, 0.70710678], "excite": [0.0, 1.0], "suppress": [-1.0, 0.0]},
}
FIELDS = ["variant", "seed", "budget", "task_seed", "valid", "success", "energy", "outcome_bin", "plan_theta",
          "filter_theta"]


def model_variants(learned: np.ndarray, fit: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, planner parameters, filter parameters) of every mixed model."""
    out = []
    for j, name in enumerate(DIAG["parameters"]):
        a = learned.copy()
        a[j] = fit[j]
        out.append((f"learned+fit:{name}", a, a))
        b = fit.copy()
        b[j] = learned[j]
        out.append((f"fit+learned:{name}", b, b))
    out.append(("plan:learned/filter:fit", learned.copy(), fit.copy()))
    out.append(("plan:fit/filter:learned", fit.copy(), learned.copy()))
    return out


def diag_jobs(runs: list[dict[str, Any]], fit: np.ndarray) -> list[dict[str, Any]]:
    from experiments.tnsre.eval_spiking_sessions import PROTOCOL

    jobs = []
    for run in sorted((r for r in runs if r["policy"] == DIAG["policy"]), key=lambda r: r["seed"]):
        learned = np.asarray(run["est"][int(DIAG["checkpoint"])], dtype=np.float64)
        for label, plan_theta, filter_theta in model_variants(learned, fit):
            for budget in DIAG["budgets"]:
                for j in range(int(PROTOCOL["task_sessions"])):
                    jobs.append({"variant": label, "run": run, "budget": float(budget),
                                 "task_seed": int(PROTOCOL["task_seeds"][DIAG["task"]]) + j,
                                 "plan_theta": plan_theta, "filter_theta": filter_theta})
    return jobs


def stage_task(out: Path, worker_index: int, n_workers: int) -> None:
    from experiments.tnsre.eval_spiking_sessions import build_network
    from experiments.tnsre.eval_spiking_sessions import IcemController, load_session_runs, reference_fit, run_task_session

    runs = load_session_runs(Path(DIAG["runs_root"]))
    fit = reference_fit()
    jobs = diag_jobs(runs, fit)[int(worker_index) :: int(n_workers)]
    path = out / "task" / f"worker_{int(worker_index):03d}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()  # sessions already written by any worker
    for other in sorted(path.parent.glob("worker_*.csv")):
        with open(other) as fh:
            done |= {(r["variant"], int(r["seed"]), float(r["budget"]), int(r["task_seed"])) for r in csv.DictReader(fh)}
    net = build_network()
    new_file = not path.exists()
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for job in jobs:
            run = job["run"]
            if (job["variant"], run["seed"], job["budget"], job["task_seed"]) in done:
                continue
            model = IcemController(job["plan_theta"], run["C"], run["b"], float(net.dt_latent), seed=job["task_seed"],
                                   dynamics_type=run["dynamics_type"], full_params=run["full_params"],
                                   min_embedding_dim=run["min_embedding_dim"], filter_theta=job["filter_theta"])
            res = run_task_session(net, DIAG["task"], job["task_seed"], job["budget"], "icem", model)
            writer.writerow({"variant": job["variant"], "seed": run["seed"], "budget": job["budget"],
                             "task_seed": job["task_seed"], "valid": res["valid"], "success": res["success"],
                             "energy": res["energy"], "outcome_bin": res["outcome_bin"],
                             "plan_theta": " ".join(f"{x:.6g}" for x in job["plan_theta"]),
                             "filter_theta": " ".join(f"{x:.6g}" for x in job["filter_theta"])})
            fh.flush()
    net.close()
    print(f"worker {worker_index}: {len(jobs)} diagnostic sessions written to {path}")


def stage_score(out: Path, main_eval: Path) -> None:
    """Success per variant and budget, with the learned and fit rows of the main evaluation."""
    rows = []
    for path in sorted((out / "task").glob("worker_*.csv")):
        with open(path) as fh:
            rows.extend(r for r in csv.DictReader(fh) if r["valid"] == "1" and float(r["budget"]) in DIAG["budgets"])
    for path in sorted((main_eval / "task").glob("worker_*.csv")):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                if r["valid"] != "1" or r["task"] != DIAG["task"] or float(r["budget"]) not in DIAG["budgets"]:
                    continue
                if r["policy"] == DIAG["policy"] and int(r["checkpoint"]) == int(DIAG["checkpoint"]):
                    rows.append({**r, "variant": "learned"})
                elif r["policy"] == "reduced_fit":
                    rows.append({**r, "variant": "fit"})
    groups: dict[tuple, dict[int, list[int]]] = {}
    for r in rows:
        groups.setdefault((r["variant"], float(r["budget"])), {}).setdefault(int(r["seed"]), []).append(int(r["success"]))
    summary = []
    for (variant, budget), per_seed in sorted(groups.items()):
        means = np.array([np.mean(v) for v in per_seed.values()])
        summary.append({"variant": variant, "budget": budget,
                        "success": float(np.mean([x for v in per_seed.values() for x in v])),
                        "sem_over_seeds": float(means.std(ddof=1) / np.sqrt(means.size)) if means.size > 1 else 0.0,
                        "n_seeds": int(means.size), "n_sessions": int(sum(len(v) for v in per_seed.values()))})
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"{len(rows)} sessions; summary in {out / 'summary.csv'}")


def stage_record(out: Path) -> None:
    """Overturn trials in the control regime; states (K, T+1, 2) and total drives (K, T, 2) from the decision."""
    from experiments.tnsre.eval_spiking_sessions import build_network
    from experiments.tnsre.eval_spiking_sessions import PROTOCOL, evidence_drive, evidence_pool

    net = build_network()
    dt = float(net.dt_latent)
    n_push = int(float(DIAG["regime_budget"]) / dt + 1e-9)
    states, inputs, meta = [], [], []
    for seed in DIAG["regime_seeds"]:
        pool = evidence_pool(seed)
        for name, direction in DIAG["regime_directions"].items():
            z = net.reset(int(seed))
            t = 0
            while t < int(PROTOCOL["max_session_bins"]) and abs(float(z[0] - z[1])) <= float(PROTOCOL["decision_gap"]):
                z = net.step(evidence_drive(t, pool))[1]
                t += 1
            if abs(float(z[0] - z[1])) <= float(PROTOCOL["decision_gap"]):
                break  # no decision: skip this seed for every direction
            target = 1 if float(z[0] - z[1]) > 0 else 0
            d = np.asarray(direction, dtype=np.float64)
            push = d if target == 1 else d[::-1]  # directions are written for target pool 2
            zs, us = [np.asarray(z, dtype=np.float64)], []
            for k in range(int(DIAG["regime_bins"])):
                u = evidence_drive(t + k, pool) + (push if k < n_push else 0.0)
                zs.append(np.asarray(net.step(u)[1], dtype=np.float64))
                us.append(u)
            states.append(np.stack(zs))
            inputs.append(np.stack(us))
            meta.append({"seed": seed, "direction": name, "onset": t, "target": target})
    net.close()
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "control_regime.npz", states=np.stack(states), inputs=np.stack(inputs))
    with open(out / "control_regime_meta.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(meta[0].keys()))
        writer.writeheader()
        writer.writerows(meta)
    print(f"{len(states)} control-regime trajectories written to {out}")


def stage_regime_r2(out: Path) -> None:
    """Held-out rollout R2 of every learned model (after 1, 5, 20 sessions) and the fit, per push direction."""
    import torch

    from actdyn.utils.validation import rollout_r2_on_trajectories
    from experiments.tnsre.eval_spiking_sessions import PROTOCOL, load_session_runs, reference_fit

    data = dict(np.load(out / "control_regime.npz"))
    with open(out / "control_regime_meta.csv") as fh:
        meta = list(csv.DictReader(fh))
    runs = load_session_runs(Path(DIAG["runs_root"]))
    fit = reference_fit()
    labels = [("fit", -1, -1)] + [(run["policy"], run["seed"], k) for run in runs for k in (1, 5, 20)]
    thetas = np.array([fit] + [run["est"][k] for run in runs for k in (1, 5, 20)])
    rows = []
    for direction in list(DIAG["regime_directions"]) + ["all"]:
        idx = [i for i, m in enumerate(meta) if direction == "all" or m["direction"] == direction]
        r2 = rollout_r2_on_trajectories(
            torch.as_tensor(thetas, dtype=torch.float32), dynamics_type=runs[0]["dynamics_type"],
            full_params=runs[0]["full_params"], min_embedding_dim=runs[0]["min_embedding_dim"],
            states=data["states"][idx], inputs=data["inputs"][idx], dt=0.05,
            horizon=int(PROTOCOL["r2_horizon"]), stride=int(PROTOCOL["r2_stride"]),
        )
        rows += [{"direction": direction, "policy": p, "seed": s, "sessions": k, "r2": float(v)}
                 for (p, s, k), v in zip(labels, r2)]
    with open(out / "control_regime_r2.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"control-regime R2 of {len(labels)} models written to {out / 'control_regime_r2.csv'}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=["task", "score", "record", "regime_r2"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-eval", type=Path, default=Path("results/tnsre/20260924_snn_sessions_m2/eval"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.stage == "task":
        stage_task(args.out, args.worker_index, args.n_workers)
    elif args.stage == "score":
        stage_score(args.out, args.main_eval)
    elif args.stage == "record":
        stage_record(args.out)
    else:
        stage_regime_r2(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
