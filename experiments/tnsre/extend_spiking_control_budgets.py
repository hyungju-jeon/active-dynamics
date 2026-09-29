"""Evaluate added overturn budgets on the original frozen session-20 models.

Use --pilot to replay saved budget-12 cases before running the new budgets.
Outputs are separate from the original experiment; failed and partial rows remain.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import eval_spiking_sessions as evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    experiment = args.experiment_dir.resolve()
    saved = json.loads((experiment / "eval/protocol.json").read_text())["protocol"]
    evaluation.PROTOCOL.update(saved)
    evaluation.PROTOCOL["reduced_fit"] = str(experiment / "reference/m2_fit.json")
    evaluation.PROTOCOL["task_checkpoints"] = [20]
    evaluation.PROTOCOL["budgets"] = {"overturn": [5.0, 15.0]}
    evaluation.TASKS = ("overturn",)
    runs = evaluation.load_session_runs(experiment / "tracks/wong_wang_snn_sessions_m2")
    units = [u for u in evaluation.controller_units(runs) if u["policy"] not in ("prior", "none")]
    if args.pilot:
        units = [u for u in units if (u["seed"] == 0 and u["policy"] in ("adaptive", "reduced_fit"))
                 or u["policy"] == "front"]
        jobs = [(i, "overturn", 12.0, 91000) for i in range(len(units))]
    else:
        jobs = evaluation.task_jobs(units)
    jobs = jobs[args.worker_index::args.n_workers]
    folder = args.out / ("pilot" if args.pilot else "task_warm")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"worker_{args.worker_index:03d}.csv"
    done = set()
    for prior in folder.glob("worker_*.csv"):
        with prior.open() as f:
            for row in csv.DictReader(f):
                done.add((row["policy"], int(row["seed"]), float(row["budget"]), int(row["task_seed"])))
    new = not path.exists()
    net = evaluation.build_network()
    try:
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=evaluation.TASK_FIELDS)
            if new:
                writer.writeheader()
            for i, task, budget, task_seed in jobs:
                unit = units[i]
                key = (unit["policy"], unit["seed"], budget, task_seed)
                if key in done:
                    continue
                start = time.monotonic()
                model = None
                if unit["theta"] is not None:
                    model = evaluation.IcemController(
                        unit["theta"], unit["C"], unit["b"], float(net.dt_latent), seed=task_seed,
                        dynamics_type=unit["dynamics_type"], full_params=unit["full_params"],
                        min_embedding_dim=unit["min_embedding_dim"], warm_start=True)
                result = evaluation.run_task_session(net, task, task_seed, budget, unit["controller"], model)
                theta = unit["theta"] if unit["theta"] is not None else [np.nan, np.nan]
                writer.writerow(dict(controller=unit["controller"], policy=unit["policy"], seed=unit["seed"],
                                     checkpoint=unit["checkpoint"], task=task, budget=budget, task_seed=task_seed,
                                     **result, w_plus=float(theta[0]), w_minus=float(theta[1]),
                                     theta=" ".join(f"{x:.6g}" for x in np.ravel(theta))))
                f.flush()
                print(f"{key}: success={result['success']}, seconds={time.monotonic()-start:.2f}", flush=True)
    finally:
        net.close()


if __name__ == "__main__":
    main()
