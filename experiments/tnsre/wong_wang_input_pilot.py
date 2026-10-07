"""Paired response-family/input pilot on the full Wang spiking network.

Fits use recorded NMDA proxies: this is a privileged model-family feasibility
check, not online PALDI identification. Each fitted family has five parameters.
Four SNN arms isolate partial current targeting, synaptic sensory evidence,
and their combination. Train/test seeds are disjoint and shared across arms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from actdyn.environment.spiking_decision import WangDecisionNetwork, calibration_input_program
from actdyn.utils.vectorfields_eqn import WongWangInsideGain, WongWangTransfer

ARMS = {
    "uniform_current": dict(stim_fraction=1., stim_weight_cv=0., evidence_mode="current"),
    "partial_current": dict(stim_fraction=.5, stim_weight_cv=.25, evidence_mode="current"),
    "synaptic_evidence": dict(stim_fraction=1., stim_weight_cv=0., evidence_mode="poisson"),
    "both": dict(stim_fraction=.5, stim_weight_cv=.25, evidence_mode="poisson"),
}
FAMILIES = {"logistic": WongWangInsideGain, "wong_wang_transfer": WongWangTransfer}
SOURCE_FILES = [
    "actdyn/utils/vectorfields_eqn.py", "actdyn/environment/spiking_decision.py",
    "actdyn/environment/vectorfield.py", "experiments/experiment_definitions.py",
    "experiments/tnsre/config/experiment_env.yaml",
    "experiments/tnsre/exp_wong_wang_inputs.py", __file__,
]


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def collect(network, seeds, *, bins: int, hold: int):
    """Record states (trials,T+1,2), controls/evidence (trials,T,2), integer counts."""
    states, controls, evidences, counts = [], [], [], []
    for seed in seeds:
        u = calibration_input_program(bins, hold_steps=hold, seed=seed)
        e = np.zeros_like(u)
        pool = int(np.random.default_rng(seed + 123).integers(2))
        e[:min(200, bins), pool] = .3  # nominal 6 pA for 1 s
        z, y = [network.reset(seed)], []
        for action, evidence in zip(u, e):
            count, state = network.step(action, evidence=evidence)
            z.append(state)
            y.append(count)
        states.append(z)
        controls.append(u)
        evidences.append(e)
        counts.append(y)
    return {"states": np.asarray(states), "controls": np.asarray(controls),
            "evidence": np.asarray(evidences), "counts": np.asarray(counts),
            "seeds": np.asarray(seeds)}


def windows(data, *, horizon: int, stride: int):
    """Create float64 rollout windows; input is nominal control + evidence.

    Both model families intentionally retain one effective input gain. A
    synaptic evidence stream is not exactly equivalent to an injected current.
    """
    z, u = data["states"], data["controls"] + data["evidence"]
    starts = range(0, u.shape[1] - horizon + 1, stride)
    z0, inputs, targets = [], [], []
    for episode in range(len(z)):
        for t in starts:
            z0.append(z[episode, t])
            inputs.append(u[episode, t:t + horizon])
            targets.append(z[episode, t + 1:t + horizon + 1])
    return tuple(torch.as_tensor(np.asarray(a), dtype=torch.float64) for a in (z0, inputs, targets))


def predict(model, theta: torch.Tensor, batch):
    """Euler rollout (starts,windows,horizon,2), with no clipping or spike corrections."""
    model._set_params(*theta.unbind(-1))
    z0, inputs, _ = batch
    z = z0.unsqueeze(0).expand(theta.shape[0], -1, -1)
    predictions = []
    for t in range(inputs.shape[1]):
        z = z + .05 * model.compute_with_input(z, inputs[None, :, t])
        predictions.append(z)
    return torch.stack(predictions, dim=2)


def fit(family, batch, *, starts: int, iterations: int, seed: int, lr: float = .025):
    """Fit all starts to the same train windows; choose solely by training MSE."""
    model = FAMILIES[family]()
    initial = model.dyn_params.detach().double().reshape(1, 5)
    rng = torch.Generator().manual_seed(seed)
    perturbation = .2 * torch.randn(starts, 5, generator=rng, dtype=torch.float64)
    perturbation[0] = 0
    theta = (initial + perturbation).requires_grad_()
    optimizer = torch.optim.Adam([theta], lr=lr)
    best = torch.full((starts,), float("inf"), dtype=torch.float64)
    best_theta = theta.detach().clone()
    history = []
    for iteration in range(iterations + 1):
        optimizer.zero_grad()
        prediction = predict(model, theta, batch)
        loss = (prediction - batch[2][None]).square().mean(dim=(1, 2, 3))
        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"{family}: nonfinite fit at iteration {iteration}: {loss.tolist()}")
        with torch.no_grad():
            improved = loss < best
            best[improved] = loss[improved]
            best_theta[improved] = theta[improved]
        history.append(loss.detach().tolist())
        if iteration < iterations:
            loss.sum().backward()
            if not torch.isfinite(theta.grad).all():
                raise FloatingPointError(f"{family}: nonfinite gradient at iteration {iteration}")
            torch.nn.utils.clip_grad_norm_([theta], 10.)
            optimizer.step()
    selected = int(best.argmin())
    return {"theta": best_theta[selected].tolist(), "selected_start": selected,
            "train_mse": float(best[selected]), "best_mse_by_start": best.tolist(),
            "initial_theta": (initial + perturbation).detach().tolist(), "loss_history": history}


def score(family, theta, batch):
    """Pool squared errors across windows and time; keep two coordinate means."""
    with torch.no_grad():
        prediction = predict(FAMILIES[family](), torch.tensor([theta], dtype=torch.float64), batch)[0]
    target = batch[2]
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite held-out prediction")
    residual = (prediction - target).square().sum()
    total = (target - target.mean(dim=(0, 1), keepdim=True)).square().sum()
    return {"r2": float(1 - residual / total),
            "rmse": float(torch.sqrt((prediction - target).square().mean())),
            "out_of_bounds_fraction": float((prediction.abs() > 2).double().mean())}, prediction.numpy()


def plot(out: Path):
    """Plot all replicate differences without treating overlapping windows as replicates."""
    from experiments.tnsre.figures.theme import apply_style, style_axis
    from experiments.tnsre.figures.assets import save_figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_style(plt)
    with (out / "scores.csv").open() as f:
        rows = list(csv.DictReader(f))
    lookup = {(r["arm"], r["replicate"], r["family"]): float(r["r2"]) for r in rows}
    reps = sorted({r["replicate"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    fig.subplots_adjust(left=.26, right=.96, bottom=.22, top=.96)
    for j, arm in enumerate(ARMS):
        differences = [lookup[arm, r, "wong_wang_transfer"] - lookup[arm, r, "logistic"] for r in reps]
        ax.scatter(differences, j + np.linspace(-.12, .12, len(reps)),
                   color="#3465a4", marker="o", s=25, zorder=3)
        ax.plot(np.median(differences), j, marker="|", color="black", markersize=14, mew=2)
    ax.axvline(0, color=".55", lw=1, ls="--")
    ax.set_yticks(range(4), ["Uniform current", "Partial current", "Synaptic evidence", "Both"])
    ax.invert_yaxis()
    ax.set_xlabel(r"Held-out $R^2$ difference: Wong–Wang transfer − logistic")
    style_axis(ax, grid_axis="x")
    save_figure(fig, out / "comparison.pdf", plt_module=plt)
    fig.savefig(out / "comparison.svg", bbox_inches=None)
    fig.savefig(out / "comparison.png", dpi=180, bbox_inches=None)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--train-trials", type=int, default=6)
    parser.add_argument("--test-trials", type=int, default=4)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--hold", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--fit-iterations", type=int, default=300)
    parser.add_argument("--fit-starts", type=int, default=4)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.plot_only:
        plot(args.out)
        return
    if not 0 < args.horizon <= args.bins or min(args.replicates, args.train_trials, args.test_trials,
            args.hold, args.stride, args.fit_iterations, args.fit_starts) < 1:
        parser.error("Need positive counts and 0 < horizon <= bins")
    if max(args.train_trials, args.test_trials) >= 1000:
        parser.error("Trial count must be < 1000 to keep seed blocks disjoint")
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {**vars(args), "out": str(args.out), "arms": ARMS,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                "fit_dtype": "float64", "network_dtype": "Brian2 default float64; saved proxies float32",
                "network": {"n_e": 1600, "n_i": 400, "sim_dt_ms": .1, "control_bin_ms": 5.,
                            "observed_per_pool": 40, "stim_gain_pa": 20., "background_rate_hz": 2400.},
                "scope": "Privileged latent-trajectory fit; not online identification or a control comparison",
                "source_sha256": {}, "status": "running"}
    import brian2
    manifest["brian2"] = brian2.__version__
    source_dir = args.out / "source"
    for file in SOURCE_FILES:
        path = Path(file).resolve()
        relative = path.relative_to(Path.cwd())
        destination = source_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest["source_sha256"][str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out / "working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "HEAD"]))
    save_json(args.out / "manifest.json", manifest)
    rows = []
    try:
        for rep in range(args.replicates):
            for arm, settings in ARMS.items():
                directory = args.out / f"replicate_{rep}" / arm
                directory.mkdir(parents=True)
                arm_start = time.monotonic()
                network = WangDecisionNetwork(**settings, stim_mask_seed=87000 + 2 * rep)
                try:
                    train_seeds = list(range(100000 + 10000 * rep, 100000 + 10000 * rep + args.train_trials))
                    test_seeds = list(range(105000 + 10000 * rep, 105000 + 10000 * rep + args.test_trials))
                    train = collect(network, train_seeds, bins=args.bins, hold=args.hold)
                    test = collect(network, test_seeds, bins=args.bins, hold=args.hold)
                    for split, data in (("train", train), ("test", test)):
                        np.savez_compressed(directory / f"{split}.npz", **data)
                    weights = network.stim_weights.copy()
                    np.save(directory / "stimulation_weights.npy", weights)
                    diagnostics = {"targeted_per_pool": np.count_nonzero(weights, axis=1).tolist(),
                                   "peak_control_pa_at_unit_action": (20 * weights.max(axis=1)).tolist(),
                                   "mean_weight_per_pool": weights.mean(axis=1).tolist(),
                                   "squared_current_ratio_per_pool": (weights ** 2).mean(axis=1).tolist(),
                                   "evidence_hz_per_neuron_at_0p3": .3 * network.evidence_rate_gain_hz,
                                   "test_mean_recorded_rate_hz": float(test["counts"].mean() / .005),
                                   "train_seeds": train_seeds, "test_seeds": test_seeds}
                    save_json(directory / "input_diagnostics.json", diagnostics)
                finally:
                    network.close()
                train_batch = windows(train, horizon=args.horizon, stride=args.stride)
                test_batch = windows(test, horizon=args.horizon, stride=args.stride)
                print(f"replicate={rep} arm={arm} data ready in {time.monotonic()-arm_start:.1f}s", flush=True)
                for family in FAMILIES:
                    fit_started = time.monotonic()
                    fitted = fit(family, train_batch, starts=args.fit_starts,
                                 iterations=args.fit_iterations, seed=91000 + rep)
                    save_json(directory / f"{family}_fit.json", fitted)
                    metrics, predictions = score(family, fitted["theta"], test_batch)
                    np.savez_compressed(directory / f"{family}_predictions.npz",
                                        predictions=predictions, targets=test_batch[2].numpy())
                    row = {"replicate": rep, "arm": arm, "family": family,
                           "train_mse": fitted["train_mse"], **metrics,
                           "fit_seconds": time.monotonic() - fit_started}
                    rows.append(row)
                    with (args.out / "scores.csv").open("w") as f:
                        writer = csv.DictWriter(f, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerows(rows)
                    print(json.dumps(row), flush=True)
        plot(args.out)
        manifest["status"] = "complete"
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest.update(elapsed_seconds=time.monotonic() - started, completed_fits=len(rows))
        save_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
