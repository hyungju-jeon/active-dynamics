"""Saved-run exploration and common local parameter-information diagnostic.

Run from the repository root:
    .venv/bin/python -m experiments.tnsre.figures.wilson_exploration

No experiments are rerun. The information calculation evaluates the manuscript's
measurement-corrected sensitivity recursion at true states and parameters, with
the saved Poisson readout and the same initial covariances for every policy.
It is an oracle-linearized diagnostic, not the agents' realized entropy change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .assets import (
    _ASSET_MATCHED_POLICIES, _apply_asset_style, _asset_baseline_policy_color,
    _asset_policy_label, save_figure,
)
from .theme import style_axis

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "results/tnsre/20260922_wilson_cowan_a2/tracks/wilson_cowan"
OUTPUT = ROOT / "results/tnsre/20260928_wilson_exploration"
WIDTH = 516 / 72.27
DOMAIN = (-4.5, 4.5)  # encloses every saved state of every compared run
GRID = 60  # 0.15 x 0.15 latent-coordinate cells
POLICIES = tuple(_ASSET_MATCHED_POLICIES)


def jacobians(z: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return drift Jacobians (...,2,2) and (...,2,4) in float64 latent units."""
    weights = np.array([[theta[0], -theta[1]], [theta[2], -theta[3]]])
    gain = expit(6 * (z @ weights.T / 4 + [-0.05, -0.35]))
    slope = 6 * gain * (1 - gain)
    jz = slope[..., :, None] * weights - np.eye(2)
    jt = np.zeros((*z.shape[:-1], 2, 4), dtype=np.float64)
    jt[..., 0, :2] = slope[..., 0, None] * z * [1, -1]
    jt[..., 1, 2:] = slope[..., 1, None] * z * [1, -1]
    return jz, jt


def coverage(z: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-run cumulative visited-cell fraction, pooled visit probability, outside fraction.

    z has shape (seed,time,2). Count sampled states, with no line interpolation,
    convex hull, smoothing, or clipping of out-of-domain states into edge cells.
    """
    n, t, _ = z.shape
    inside = np.all((z >= DOMAIN[0]) & (z <= DOMAIN[1]), axis=-1)
    cells = np.floor((z - DOMAIN[0]) / (DOMAIN[1] - DOMAIN[0]) * bins).astype(int)
    cells = np.clip(cells, 0, bins - 1)
    ids = cells[..., 1] * bins + cells[..., 0]
    seen = np.zeros((n, bins * bins), dtype=bool)
    curve = np.zeros((n, t))
    for k in range(t):
        rows = np.flatnonzero(inside[:, k])
        seen[rows, ids[rows, k]] = True
        curve[:, k] = seen.mean(axis=1) * 100
    return curve, seen.mean(axis=0).reshape(bins, bins) * 100, float(1 - inside.mean())


def parameter_information(z: np.ndarray, meta: dict) -> np.ndarray:
    """Common local Gaussian information proxy (nats), shape (seed,time).

    S^- = Fz S^+ + Ftheta; P^- = Fz P^+ Fz.T + Q.
    A = (Iz^-1 + P^-)^-1; Lambda += S^-.T A S^-.
    S^+ = (I - P^- A) S^-; P^+ = P^- - P^- A P^-.
    Gain = logdet(I + Ptheta0 Lambda)/2, Ptheta0=I.
    Initial state covariance is I and initial sensitivity is zero. No parameter
    diffusion, online parameter estimates, shrinkage, or policy-specific schedule
    is used. The last saved state has no successor; use the 1,999 saved pairs.
    Boundary derivatives are omitted, as in the manuscript's local Jacobians.
    """
    n, t, _ = z.shape
    dt, q = float(meta["dt"]), float(meta["state_noise"])
    c = np.asarray(meta["observation_loading_matrix"], dtype=np.float64)
    b = np.asarray(meta["observation_loading_bias"], dtype=np.float64)
    jz, jt = jacobians(z[:, :-1], np.asarray(meta["embedding_true"]))
    counts = dt * np.exp(z[:, 1:] @ c.T + b)
    iz = np.einsum("ntk,ki,kj->ntij", counts, c, c)
    sensitivity = np.zeros((n, 2, 4))
    covariance = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    precision_gain = np.zeros((n, 4, 4))
    out = np.zeros((n, t))
    for k in range(t - 1):
        fz, ft = np.eye(2) + dt * jz[:, k], dt * jt[:, k]
        predicted_s = fz @ sensitivity + ft
        predicted_p = fz @ covariance @ fz.swapaxes(-1, -2) + q * dt * np.eye(2)
        attenuation = np.linalg.inv(np.linalg.inv(iz[:, k]) + predicted_p)
        precision_gain += predicted_s.swapaxes(-1, -2) @ attenuation @ predicted_s
        correction = np.eye(2) - predicted_p @ attenuation
        sensitivity = correction @ predicted_s
        covariance = correction @ predicted_p
        covariance = (covariance + covariance.swapaxes(-1, -2)) / 2
        sign, logdet = np.linalg.slogdet(np.eye(4) + precision_gain)
        if not np.all(sign > 0):
            raise ValueError("Nonpositive diagnostic precision determinant")
        out[:, k + 1] = logdet / 2
    if not np.isfinite(out).all() or np.min(np.diff(out, axis=1)) < -1e-9:
        raise ValueError("Information diagnostic must be finite and nondecreasing")
    return out


def load_runs(source: Path, policy: str) -> tuple[np.ndarray, dict, list[dict]]:
    """Require exactly the 100 complete main-comparison runs and consecutive samples."""
    trajectories, provenance, metadata = [], [], []
    for seed in range(100):
        paths = list((source / policy / f"seed_{seed}").glob("repeat_*/run_metadata.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one run for {policy}, seed {seed}: {paths}")
        path = paths[0]
        meta = json.loads(path.read_text())
        if meta["status"] != "completed" or meta["total_steps"] != 2000 or meta["nan_detected"]:
            raise ValueError(f"Incomplete or failed run: {path}")
        trace = path.parent / "state_action_trace.csv"
        table = pd.read_csv(trace, usecols=["step", "true_z0", "true_z1"])
        np.testing.assert_array_equal(table["step"], np.arange(1, 2001))
        z = table[["true_z0", "true_z1"]].to_numpy(dtype=np.float64)
        if not np.isfinite(z).all():
            raise ValueError(f"Nonfinite trajectory: {trace}")
        trajectories.append(z)
        metadata.append(meta)
        provenance.append({"policy": policy, "seed": seed,
                           "trace": str(trace.relative_to(ROOT)),
                           "sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                           "metadata_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    common_keys = ("dt", "state_noise", "embedding_true", "observation_loading_matrix",
                   "observation_loading_bias", "parameter_prior_covariance", "state_init_uncertainty")
    for meta in metadata:
        for key in common_keys:
            np.testing.assert_allclose(meta[key], metadata[0][key])
    return np.stack(trajectories), metadata[0], provenance


def plot(data: dict, output: Path) -> None:
    """Six comparable visit maps above coverage and information learning curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _apply_asset_style(plt)
    fig = plt.figure(figsize=(WIDTH, 3.65))
    map_left, size, gap = 0.055, 0.13, 0.021
    first_map = None
    for j, policy in enumerate(POLICIES):
        ax = fig.add_axes([map_left + j * (size + gap), 0.63, size, size * WIDTH / 3.65],
                          sharey=first_map)
        if first_map is None:
            first_map = ax
        im = ax.imshow(data[policy]["map"], origin="lower", extent=(*DOMAIN, *DOMAIN),
                       vmin=0, vmax=100, cmap="cividis", interpolation="nearest")
        ax.set_title(_asset_policy_label(policy), pad=4)
        ax.set_xticks([-4, 0, 4]); ax.set_yticks([-4, 0, 4])
        ax.set_xlabel(r"$z_1$ (E)", labelpad=1)
        if j == 0:
            ax.set_ylabel(r"$z_2$ (I)", labelpad=1)
        else:
            ax.tick_params(labelleft=False)
        ax.tick_params(length=2, pad=1)
    cax = fig.add_axes([0.949, 0.63, 0.009, size * WIDTH / 3.65])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 50, 100])
    cb.ax.tick_params(length=2, pad=1, labelsize=6)
    cb.set_label("Runs (%)", rotation=0, ha="right", va="bottom", fontsize=6)
    cb.ax.yaxis.set_label_coords(1, 1.045)
    fig.text(0.5, 0.985, "Runs visiting each cell (%)", ha="center", va="top", fontsize=8)
    fig.text(0.012, 0.94, "A", fontsize=10, weight="bold")
    axes = [fig.add_axes([0.09, 0.12, 0.36, 0.33]), fig.add_axes([0.58, 0.12, 0.36, 0.33])]
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (2, 1))]
    markers = ["o", "s", "^", "D", "v", "x"]
    for j, policy in enumerate(POLICIES):
        for ax, key in zip(axes, ["coverage", "information"]):
            curves = data[policy][key]
            q = np.quantile(curves, [0.25, 0.5, 0.75], axis=0)
            idx = np.unique(np.r_[np.arange(0, curves.shape[1], 25), curves.shape[1] - 1])
            color = _asset_baseline_policy_color(policy)
            ax.fill_between(idx, q[0, idx], q[2, idx], color=color, alpha=0.10, linewidth=0)
            ax.plot(idx, q[1, idx], color=color, ls=styles[j], lw=1.1,
                    marker=markers[j], markevery=16, ms=2.4,
                    label=_asset_policy_label(policy))
    for ax in axes:
        ax.set_xlim(0, 2000); ax.set_ylim(bottom=0)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_xlabel("Environment steps", labelpad=2)
        style_axis(ax, grid_axis="y")
    axes[0].set_ylabel("Visited cells (%)", labelpad=3)
    axes[1].set_ylabel("Parameter-information\nproxy (nats)", labelpad=3)
    fig.text(0.012, 0.46, "B", fontsize=10, weight="bold")
    fig.text(0.505, 0.46, "C", fontsize=10, weight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.51, 0.515), ncol=6,
               frameon=False, columnspacing=1.2, handlelength=2.6)
    # Fixed physical size; shared exporter rejects out-of-canvas text.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches=None)
    save_figure(fig, output, plt_module=plt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, ledger, rows, summary = {}, [], [], {}
    common = None
    for policy in POLICIES:
        z, meta, provenance = load_runs(args.source, policy)
        if common is not None:
            for key in ("dt", "state_noise", "embedding_true", "observation_loading_matrix",
                        "observation_loading_bias", "parameter_prior_covariance", "state_init_uncertainty"):
                np.testing.assert_allclose(meta[key], common[key])
        common = meta
        if meta["parameter_prior_covariance"] != 1 or meta["state_init_uncertainty"] != 1:
            raise ValueError("Diagnostic initialization assumes identity covariances")
        cov, visit, outside = coverage(z, GRID)
        info = parameter_information(z, meta)
        data[policy] = {"coverage": cov, "map": visit, "information": info}
        ledger.extend(provenance)
        summary[policy] = {"seeds": 100, "outside_domain_fraction": outside,
                           "coverage_final_q25_median_q75": np.quantile(cov[:, -1], [.25, .5, .75]).tolist(),
                           "information_final_q25_median_q75": np.quantile(info[:, -1], [.25, .5, .75]).tolist(),
                           "coverage_resolution_check": {str(n): float(np.median(coverage(z, n)[0][:, -1]))
                                                         for n in [40, 80]}}
        for seed in range(100):
            for k in np.unique(np.r_[np.arange(0, 2000, 25), 1999]):
                rows.append({"policy": policy, "seed": seed, "step": int(k),
                             "coverage_percent": cov[seed, k], "information_proxy_nats": info[seed, k]})
        print(policy, json.dumps(summary[policy]), flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output_dir / "provenance.json").write_text(json.dumps({
        "source": str(args.source), "grid": GRID, "domain": DOMAIN,
        "samples_per_run": 2000, "transitions_per_run": 1999,
        "information": "oracle-linearized Gaussian/Poisson sensitivity proxy; not realized posterior gain",
        "inputs": ledger,
    }, indent=2) + "\n")
    plot(data, args.output_dir / "appendix_wilson_exploration.pdf")


if __name__ == "__main__":
    main()
