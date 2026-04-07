#!/usr/bin/env python3
"""Interactive COSYNE information-map GUI with a fixed 2x3 layout."""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LogNorm
from matplotlib.image import AxesImage
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
import numpy as np
import torch

from actdyn.environment.observation import LogLinearObservation


DYNAMICS_CHOICES = (
    "duffing",
    "van_der_pol",
    "snowman",
    "double_limit_cycle",
    "fitzhugh_nagumo",
    "hopf",
)

_DYNAMICS_ALIASES = {
    "duffing": "duffing",
    "van der pol": "van_der_pol",
    "van der poll": "van_der_pol",
    "van_der_pol": "van_der_pol",
    "van_der_poll": "van_der_pol",
    "snowman": "snowman",
    "double limit cycle": "double_limit_cycle",
    "double_limit_cycle": "double_limit_cycle",
    "fitzhugh nagumo": "fitzhugh_nagumo",
    "fitzhugh_nagumo": "fitzhugh_nagumo",
    "hopf": "hopf",
}

_DYNAMICS_LABELS = {
    "duffing": "duffing",
    "van_der_pol": "van der pol",
    "snowman": "snowman",
    "double_limit_cycle": "double limit cycle",
    "fitzhugh_nagumo": "fitzhugh nagumo",
    "hopf": "hopf",
}

_LABEL_TO_DYNAMICS = {label: key for key, label in _DYNAMICS_LABELS.items()}

_DYNAMICS_PRESETS = {
    "duffing": {
        "a_label": "a",
        "a_min": -3.0,
        "a_max": -0.1,
        "a_init": -1.55,
        "b_label": "b",
        "b_min": -2.0,
        "b_max": 2.0,
        "b_init": 0.0,
    },
    "van_der_pol": {
        "a_label": "mu",
        "a_min": 0.1,
        "a_max": 3.0,
        "a_init": 1.0,
        "b_label": "w",
        "b_min": 0.1,
        "b_max": 2.0,
        "b_init": 1.0,
    },
    "snowman": {
        "a_label": "w",
        "a_min": 0.1,
        "a_max": 2.0,
        "a_init": 1.0,
        "b_label": "d",
        "b_min": 0.2,
        "b_max": 2.0,
        "b_init": 1.0,
    },
    "double_limit_cycle": {
        "a_label": "w",
        "a_min": 0.1,
        "a_max": 2.0,
        "a_init": 1.0,
        "b_label": "d",
        "b_min": 0.2,
        "b_max": 2.0,
        "b_init": 1.0,
    },
    "fitzhugh_nagumo": {
        "a_label": "i_ext",
        "a_min": -0.5,
        "a_max": 1.5,
        "a_init": 0.5,
        "b_label": "a",
        "b_min": 0.2,
        "b_max": 1.2,
        "b_init": 0.7,
    },
    "hopf": {
        "a_label": "omega",
        "a_min": 0.1,
        "a_max": 2.5,
        "a_init": 1.2,
        "b_label": "mu",
        "b_min": -1.0,
        "b_max": 1.0,
        "b_init": 0.3,
    },
}

_DEFAULT_SLIDER_A_MIN = -3.0
_DEFAULT_SLIDER_A_MAX = -0.1
_DEFAULT_SLIDER_B_MIN = -2.0
_DEFAULT_SLIDER_B_MAX = 2.0


def normalize_dynamics_type(raw: str) -> str:
    """Normalize user-provided dynamics name to canonical key."""

    key = raw.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    if key in _DYNAMICS_ALIASES:
        return _DYNAMICS_ALIASES[key]
    raise ValueError(
        "Unknown dynamics type: "
        f"{raw}. Expected one of: {', '.join(sorted(_DYNAMICS_LABELS.values()))}"
    )


def _parse_dynamics_type(raw: str) -> str:
    """Argparse adapter for dynamics type parsing."""

    try:
        return normalize_dynamics_type(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def get_dynamics_preset(dynamics_type: str) -> Dict[str, object]:
    """Return slider preset metadata for a dynamics choice."""

    return _DYNAMICS_PRESETS[normalize_dynamics_type(dynamics_type)]


def resolve_slider_configuration(
    *,
    dynamics_type: str,
    slider_a_min: Optional[float],
    slider_a_max: Optional[float],
    slider_b_min: Optional[float],
    slider_b_max: Optional[float],
) -> Dict[str, object]:
    """Resolve slider bounds and initial values for the selected dynamics."""

    dynamics_key = normalize_dynamics_type(dynamics_type)
    preset = dict(get_dynamics_preset(dynamics_type))
    using_parser_defaults = (
        slider_a_min == _DEFAULT_SLIDER_A_MIN
        and slider_a_max == _DEFAULT_SLIDER_A_MAX
        and slider_b_min == _DEFAULT_SLIDER_B_MIN
        and slider_b_max == _DEFAULT_SLIDER_B_MAX
    )
    if dynamics_key != "duffing" and using_parser_defaults:
        return preset

    a_min = float(preset["a_min"]) if slider_a_min is None else float(slider_a_min)
    a_max = float(preset["a_max"]) if slider_a_max is None else float(slider_a_max)
    b_min = float(preset["b_min"]) if slider_b_min is None else float(slider_b_min)
    b_max = float(preset["b_max"]) if slider_b_max is None else float(slider_b_max)
    preset["a_min"] = a_min
    preset["a_max"] = a_max
    preset["b_min"] = b_min
    preset["b_max"] = b_max
    preset["a_init"] = float(preset["a_init"]) if slider_a_min is None and slider_a_max is None else 0.5 * (
        a_min + a_max
    )
    preset["b_init"] = float(preset["b_init"]) if slider_b_min is None and slider_b_max is None else 0.5 * (
        b_min + b_max
    )
    return preset


@dataclass
class GuiState:
    """Container for interactive GUI state."""

    c_raw: np.ndarray
    c: np.ndarray
    bias: np.ndarray
    a: float
    b_param: float
    asymmetric: bool
    z_flat: np.ndarray
    x_grid: np.ndarray
    v_grid: np.ndarray
    grid_lim: float
    n_grid: int
    mean_firing: float
    eps_det: float
    dt: float
    obs_dim: int
    dynamics_type: str
    rng: Optional[np.random.Generator] = None


@dataclass
class MapBundle:
    """Collection of map tensors and scalar diagnostics for rendering."""

    lambda_mean_map: np.ndarray
    iz_logdet_map: np.ndarray
    ss_logdet_map: np.ndarray
    itheta_logdet_map: np.ndarray
    requested_diag: Dict[str, float]
    current_code_diag: Dict[str, float]


def apply_loading_asymmetry(c_raw: np.ndarray, asymmetric: bool) -> np.ndarray:
    """Apply COSYNE asymmetric loading transform to raw observation weights."""

    c = np.asarray(c_raw, dtype=np.float64).copy()
    if asymmetric:
        c[:, 0] = np.abs(c[:, 0])
        c[:, 1] = 2.0 * c[:, 1]
    return c


def compute_loading_bias(c: np.ndarray, mean_firing: float) -> np.ndarray:
    """Compute bias as log(mean_firing) - 0.5*diag(CC^T)."""

    c = np.asarray(c, dtype=np.float64)
    mean_term = np.log(float(mean_firing) * np.ones(c.shape[0], dtype=np.float64))
    return mean_term - 0.5 * np.diag(c @ c.T)


def compute_lambda_values(c: np.ndarray, bias: np.ndarray, z_flat: np.ndarray) -> np.ndarray:
    """Compute lambda(z)=exp(Cz+b) on a flattened latent grid."""

    linear = z_flat @ c.T + bias[None, :]
    linear = np.clip(linear, -60.0, 60.0)
    return np.exp(linear)


def aggregate_lambda_mean(lambda_values: np.ndarray) -> np.ndarray:
    """Aggregate lambda vector per point by mean firing rate."""

    return np.mean(lambda_values, axis=1)


def compute_iz_matrices(lambda_values: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute I_z(z)=C^T diag(lambda(z)) C on all grid points."""

    weighted_c = lambda_values[:, :, None] * c[None, :, :]
    i_z = np.einsum("ndk,dl->nkl", weighted_c, c, optimize=True)
    return 0.5 * (i_z + np.swapaxes(i_z, -1, -2))


def compute_duffing_theta_sensitivity(z_flat: np.ndarray) -> np.ndarray:
    """Compute F_theta(z)=[[0,0],[v,-x]] for each flattened grid point."""

    x = z_flat[:, 0]
    v = z_flat[:, 1]
    f_theta = np.zeros((z_flat.shape[0], 2, 2), dtype=np.float64)
    f_theta[:, 1, 0] = v
    f_theta[:, 1, 1] = -x
    return f_theta


def compute_dynamics_velocity(
    z_flat: np.ndarray,
    *,
    dynamics_type: str,
    param_a: float,
    param_b: float,
) -> np.ndarray:
    """Compute vector-field velocity for selected dynamics on flattened grid."""

    dynamics_key = normalize_dynamics_type(dynamics_type)
    x = z_flat[:, 0]
    v = z_flat[:, 1]
    a_val = float(param_a)
    b_val = float(param_b)

    if dynamics_key == "duffing":
        dx = v
        dv = a_val * v - b_val * x - 0.1 * x**3
    elif dynamics_key == "van_der_pol":
        dx = v
        dv = a_val * (1.0 - x**2) * v - b_val * x
    elif dynamics_key == "fitzhugh_nagumo":
        b_fixed = 0.8
        tau_fixed = 12.5
        dx = x - x**3 / 3.0 - v + a_val
        dv = (x + b_val - b_fixed * v) / tau_fixed
    elif dynamics_key == "hopf":
        radius_sq = x**2 + v**2
        dx = (b_val - radius_sq) * x - a_val * v
        dv = a_val * x + (b_val - radius_sq) * v
    elif dynamics_key == "double_limit_cycle":
        r = np.sqrt(x**2 + v**2)
        dx = x * (b_val - r) - a_val * v * (2.0 * b_val - r)
        dv = v * (b_val - r) + a_val * x * (2.0 * b_val - r)
    elif dynamics_key == "snowman":
        beta = 10.0
        r1 = np.sqrt((x - b_val) ** 2 + v**2)
        u1 = (x - b_val) * (b_val**2 - r1**2) - a_val * v
        w1 = v * (b_val**2 - r1**2) + a_val * (x - b_val)

        r2 = np.sqrt((x + b_val) ** 2 + v**2)
        u2 = (x + b_val) * (b_val**2 - r2**2) + a_val * v
        w2 = v * (b_val**2 - r2**2) - a_val * (x + b_val)

        sigma_pos = 1.0 / (1.0 + np.exp(-np.clip(beta * x, -60.0, 60.0)))
        sigma_neg = 1.0 - sigma_pos
        dx = sigma_pos * u1 + sigma_neg * u2
        dv = sigma_pos * w1 + sigma_neg * w2
    else:
        raise ValueError(f"Unsupported dynamics_type after normalization: {dynamics_key}")

    velocity = np.stack([dx, dv], axis=1)
    return np.nan_to_num(velocity, nan=0.0, posinf=1e6, neginf=-1e6)


def compute_theta_sensitivity(
    z_flat: np.ndarray,
    *,
    dynamics_type: str,
    param_a: float,
    param_b: float,
    fd_step: float = 1e-4,
) -> np.ndarray:
    """Compute F_theta by centered finite differences of selected dynamics."""

    h = float(fd_step)
    vel_ap = compute_dynamics_velocity(
        z_flat,
        dynamics_type=dynamics_type,
        param_a=float(param_a) + h,
        param_b=float(param_b),
    )
    vel_am = compute_dynamics_velocity(
        z_flat,
        dynamics_type=dynamics_type,
        param_a=float(param_a) - h,
        param_b=float(param_b),
    )
    vel_bp = compute_dynamics_velocity(
        z_flat,
        dynamics_type=dynamics_type,
        param_a=float(param_a),
        param_b=float(param_b) + h,
    )
    vel_bm = compute_dynamics_velocity(
        z_flat,
        dynamics_type=dynamics_type,
        param_a=float(param_a),
        param_b=float(param_b) - h,
    )

    f_theta = np.zeros((z_flat.shape[0], 2, 2), dtype=np.float64)
    f_theta[:, :, 0] = (vel_ap - vel_am) / (2.0 * h)
    f_theta[:, :, 1] = (vel_bp - vel_bm) / (2.0 * h)
    return f_theta


def compute_ss_matrices(f_theta: np.ndarray) -> np.ndarray:
    """Compute SS(z)=F_theta(z)^T F_theta(z)."""

    ss = np.einsum("nki,nkj->nij", f_theta, f_theta, optimize=True)
    return 0.5 * (ss + np.swapaxes(ss, -1, -2))


def _batched_solve(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Solve left @ X = right with pseudoinverse fallback."""

    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        inv_left = np.linalg.pinv(left)
        return np.einsum("nij,njk->nik", inv_left, right, optimize=True)


def compute_itheta_matrices(i_z: np.ndarray, f_theta: np.ndarray) -> np.ndarray:
    """Compute I_theta(z)=F_theta(z)^T I_z(z) F_theta(z)."""

    i_theta = np.einsum("nki,nkl,nlj->nij", f_theta, i_z, f_theta, optimize=True)
    return 0.5 * (i_theta + np.swapaxes(i_theta, -1, -2))


def compute_logdet_map(matrices: np.ndarray, eps: float) -> np.ndarray:
    """Compute stable logdet(M + eps*I) pointwise over a matrix field."""

    dim = matrices.shape[-1]
    eye = np.eye(dim, dtype=np.float64)[None, :, :]
    reg = 0.5 * (matrices + np.swapaxes(matrices, -1, -2)) + float(eps) * eye
    sign, logabsdet = np.linalg.slogdet(reg)
    fallback = np.log(max(float(eps), 1e-16))
    out = np.where(sign > 0.0, logabsdet, fallback)
    return np.nan_to_num(out, nan=fallback, posinf=1e6, neginf=-1e6)


def sample_raw_loading_matrix(
    obs_dim: int,
    *,
    latent_dim: int = 2,
    dt: float = 0.01,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Sample raw loading matrix from the observation-model initializer."""

    if seed is None:
        obs_model = LogLinearObservation(
            d_obs=obs_dim,
            d_latent=latent_dim,
            noise_type="poisson",
            dt=float(dt),
            device="cpu",
        )
        return obs_model.network[0].weight.detach().cpu().numpy().astype(np.float64)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        obs_model = LogLinearObservation(
            d_obs=obs_dim,
            d_latent=latent_dim,
            noise_type="poisson",
            dt=float(dt),
            device="cpu",
        )
    return obs_model.network[0].weight.detach().cpu().numpy().astype(np.float64)


def _reshape_map(values_flat: np.ndarray, n_grid: int) -> np.ndarray:
    """Reshape flattened scalar field into (n_grid, n_grid)."""

    return values_flat.reshape(n_grid, n_grid)


def _trace_mean(matrices: np.ndarray) -> float:
    """Compute spatial mean of matrix trace for a matrix field."""

    trace_vals = np.trace(matrices, axis1=-2, axis2=-1)
    return float(np.mean(trace_vals))


def compute_current_code_diagnostics(
    *,
    c: np.ndarray,
    bias: np.ndarray,
    z_flat: np.ndarray,
    dt: float,
    dynamics_type: str,
    param_a: float,
    param_b: float,
) -> Dict[str, float]:
    """Compute scalar diagnostics under current-code I_z/I_theta convention."""

    lambda_requested = compute_lambda_values(c=c, bias=bias, z_flat=z_flat)
    lambda_code = float(dt) * lambda_requested
    i_z_code = compute_iz_matrices(lambda_values=lambda_code, c=c)

    f_theta = compute_theta_sensitivity(
        z_flat,
        dynamics_type=dynamics_type,
        param_a=param_a,
        param_b=param_b,
    )
    eye = np.eye(2, dtype=np.float64)[None, :, :]
    atten = _batched_solve(eye + i_z_code, i_z_code)
    i_theta_code = compute_itheta_matrices(i_z=atten, f_theta=f_theta)

    return {
        "iz_trace_mean": _trace_mean(i_z_code),
        "itheta_trace_mean": _trace_mean(i_theta_code),
    }


def compute_map_bundle(state: GuiState) -> MapBundle:
    """Compute all requested map panels and scalar diagnostics."""

    lambda_values = compute_lambda_values(c=state.c, bias=state.bias, z_flat=state.z_flat)
    lambda_mean = np.nan_to_num(aggregate_lambda_mean(lambda_values), nan=0.0, posinf=1e6, neginf=0.0)

    i_z = compute_iz_matrices(lambda_values=lambda_values, c=state.c)
    f_theta = compute_theta_sensitivity(
        state.z_flat,
        dynamics_type=state.dynamics_type,
        param_a=state.a,
        param_b=state.b_param,
    )
    ss = compute_ss_matrices(f_theta=f_theta)
    i_theta = compute_itheta_matrices(i_z=i_z, f_theta=f_theta)

    iz_logdet = compute_logdet_map(i_z, eps=state.eps_det)
    ss_logdet = compute_logdet_map(ss, eps=state.eps_det)
    itheta_logdet = compute_logdet_map(i_theta, eps=state.eps_det)

    requested_diag = {
        "lambda_mean": float(np.mean(lambda_mean)),
        "iz_logdet_mean": float(np.mean(iz_logdet)),
        "ss_logdet_mean": float(np.mean(ss_logdet)),
        "itheta_logdet_mean": float(np.mean(itheta_logdet)),
    }
    current_code_diag = compute_current_code_diagnostics(
        c=state.c,
        bias=state.bias,
        z_flat=state.z_flat,
        dt=state.dt,
        dynamics_type=state.dynamics_type,
        param_a=state.a,
        param_b=state.b_param,
    )

    return MapBundle(
        lambda_mean_map=_reshape_map(lambda_mean, state.n_grid),
        iz_logdet_map=_reshape_map(iz_logdet, state.n_grid),
        ss_logdet_map=_reshape_map(ss_logdet, state.n_grid),
        itheta_logdet_map=_reshape_map(itheta_logdet, state.n_grid),
        requested_diag=requested_diag,
        current_code_diag=current_code_diag,
    )


def _setup_map_axis(ax: Axes, title: str, grid_lim: float) -> None:
    """Configure map axis style shared across panels."""

    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_title(title)


def _draw_vector_field(
    ax: Axes,
    *,
    dynamics_type: str,
    a: float,
    b_param: float,
    grid_lim: float,
    n_grid: int = 31,
) -> None:
    """Render vector field panel for the current dynamics and parameters."""

    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float64)
    x_grid, v_grid = np.meshgrid(axis, axis, indexing="xy")
    z_flat = np.stack([x_grid.ravel(), v_grid.ravel()], axis=1)
    velocity = compute_dynamics_velocity(
        z_flat,
        dynamics_type=dynamics_type,
        param_a=a,
        param_b=b_param,
    )
    dx = velocity[:, 0].reshape(x_grid.shape)
    dv = velocity[:, 1].reshape(v_grid.shape)
    speed = np.sqrt(dx**2 + dv**2)

    ax.clear()
    ax.streamplot(
        x_grid,
        v_grid,
        dx,
        dv,
        color=speed,
        linewidth=0.7,
        density=1.5,
        cmap="viridis",
    )
    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    dyn_label = _DYNAMICS_LABELS[normalize_dynamics_type(dynamics_type)]
    ax.set_title(f"Vector Field [{dyn_label}] (a={a:.3f}, b={b_param:.3f})")


def _format_diagnostics(state: GuiState, bundle: MapBundle) -> str:
    """Format control-panel diagnostics text block."""

    req = bundle.requested_diag
    code = bundle.current_code_diag
    lines = [
        "Requested-equation summaries",
        f"  mean(lambda): {req['lambda_mean']:.4e}",
        f"  mean logdet(I_z): {req['iz_logdet_mean']:.4e}",
        f"  mean logdet(SS): {req['ss_logdet_mean']:.4e}",
        f"  mean logdet(I_theta): {req['itheta_logdet_mean']:.4e}",
        "",
        "Current-code diagnostics",
        f"  mean trace(I_z): {code['iz_trace_mean']:.4e}",
        f"  mean trace(I_theta): {code['itheta_trace_mean']:.4e}",
    ]
    return "\n".join(lines)


def _create_grid(grid_lim: float, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create flattened and mesh latent grid arrays."""

    axis = np.linspace(-grid_lim, grid_lim, n_grid, dtype=np.float64)
    x_grid, v_grid = np.meshgrid(axis, axis, indexing="xy")
    z_flat = np.stack([x_grid.ravel(), v_grid.ravel()], axis=1)
    return z_flat, x_grid, v_grid


def _finite_limits(values: np.ndarray) -> tuple[float, float]:
    """Pick robust finite color limits for panel updates."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.0))
    if not np.isfinite(vmin):
        vmin = float(np.min(finite))
    if not np.isfinite(vmax):
        vmax = float(np.max(finite))
    if vmax <= vmin:
        delta = max(1e-6, 0.01 * max(1.0, abs(vmin)))
        vmax = vmin + delta
    return vmin, vmax


def _log_norm_from_positive(values: np.ndarray) -> LogNorm:
    """Pick robust LogNorm for positive-valued maps."""

    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size == 0:
        return LogNorm(vmin=1e-8, vmax=1.0)
    vmin = float(np.percentile(finite_positive, 1.0))
    vmax = float(np.percentile(finite_positive, 99.0))
    vmin = max(vmin, 1e-12)
    if vmax <= vmin:
        vmax = vmin * 1.01
    return LogNorm(vmin=vmin, vmax=vmax)


def _update_image(image: AxesImage, colorbar: Colorbar, values: np.ndarray) -> None:
    """Update panel image and refresh colorbar with linear normalization."""

    image.set_data(values)
    vmin, vmax = _finite_limits(values)
    image.set_clim(vmin=vmin, vmax=vmax)
    colorbar.update_normal(image)


def _update_image_log(image: AxesImage, colorbar: Colorbar, values: np.ndarray) -> None:
    """Update panel image and refresh colorbar with log normalization."""

    image.set_data(values)
    image.set_norm(_log_norm_from_positive(values))
    colorbar.update_normal(image)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for GUI launch configuration."""

    parser = argparse.ArgumentParser(description="Interactive COSYNE information-map GUI")
    parser.add_argument("--grid-lim", type=float, default=10.0)
    parser.add_argument("--n-grid", type=int, default=101)
    parser.add_argument("--obs-dim", type=int, default=50)
    parser.add_argument("--mean-firing", type=float, default=50.0)
    parser.add_argument("--eps-det", type=float, default=1e-8)
    parser.add_argument("--slider-a-min", type=float, default=_DEFAULT_SLIDER_A_MIN)
    parser.add_argument("--slider-a-max", type=float, default=_DEFAULT_SLIDER_A_MAX)
    parser.add_argument("--slider-b-min", type=float, default=_DEFAULT_SLIDER_B_MIN)
    parser.add_argument("--slider-b-max", type=float, default=_DEFAULT_SLIDER_B_MAX)
    parser.add_argument(
        "--dynamics-type",
        type=_parse_dynamics_type,
        default="duffing",
        help=(
            "Dynamics selector: "
            + ", ".join(_DYNAMICS_LABELS[key] for key in DYNAMICS_CHOICES[:-1])
            + ", or "
            + _DYNAMICS_LABELS[DYNAMICS_CHOICES[-1]]
            + "."
        ),
    )
    parser.add_argument(
        "--asymmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply asymmetric loading transform C[:,0]=abs(C[:,0]), C[:,1]=2*C[:,1].",
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Launch the interactive GUI."""

    args = build_parser().parse_args(argv)

    n_grid = max(11, int(args.n_grid))
    grid_lim = float(args.grid_lim)
    dt = 0.01
    slider_config = resolve_slider_configuration(
        dynamics_type=str(args.dynamics_type),
        slider_a_min=args.slider_a_min,
        slider_a_max=args.slider_a_max,
        slider_b_min=args.slider_b_min,
        slider_b_max=args.slider_b_max,
    )
    rng = np.random.default_rng(args.seed) if args.seed is not None else None
    initial_seed = int(rng.integers(0, 2**31 - 1)) if rng is not None else None
    c_raw = sample_raw_loading_matrix(obs_dim=int(args.obs_dim), dt=dt, seed=initial_seed)
    c = apply_loading_asymmetry(c_raw, asymmetric=bool(args.asymmetric))
    bias = compute_loading_bias(c, mean_firing=float(args.mean_firing))
    z_flat, x_grid, v_grid = _create_grid(grid_lim=grid_lim, n_grid=n_grid)

    state = GuiState(
        c_raw=c_raw,
        c=c,
        bias=bias,
        a=float(slider_config["a_init"]),
        b_param=float(slider_config["b_init"]),
        asymmetric=bool(args.asymmetric),
        z_flat=z_flat,
        x_grid=x_grid,
        v_grid=v_grid,
        grid_lim=grid_lim,
        n_grid=n_grid,
        mean_firing=float(args.mean_firing),
        eps_det=float(args.eps_det),
        dt=dt,
        obs_dim=int(args.obs_dim),
        dynamics_type=str(args.dynamics_type),
        rng=rng,
    )

    bundle = compute_map_bundle(state)
    extent = [-grid_lim, grid_lim, -grid_lim, grid_lim]

    fig = plt.figure(figsize=(16.0, 9.2), dpi=120)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], height_ratios=[1.0, 1.0])

    ax_lambda = fig.add_subplot(gs[0, 0])
    ax_iz = fig.add_subplot(gs[0, 1])
    ax_ctrl = fig.add_subplot(gs[0, 2])
    ax_ss = fig.add_subplot(gs[1, 0])
    ax_itheta = fig.add_subplot(gs[1, 1])
    ax_vf = fig.add_subplot(gs[1, 2])

    im_lambda = ax_lambda.imshow(
        bundle.lambda_mean_map,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        norm=_log_norm_from_positive(bundle.lambda_mean_map),
    )
    im_iz = ax_iz.imshow(
        bundle.iz_logdet_map,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    im_ss = ax_ss.imshow(
        bundle.ss_logdet_map,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap="cividis",
    )
    im_itheta = ax_itheta.imshow(
        bundle.itheta_logdet_map,
        extent=extent,
        origin="lower",
        interpolation="nearest",
        cmap="plasma",
    )

    _setup_map_axis(ax_lambda, r"$\lambda$ map", grid_lim=grid_lim)
    _setup_map_axis(ax_iz, r"$\log\det(I_{z,t}+\epsilon I)$", grid_lim=grid_lim)
    _setup_map_axis(ax_ss, r"$\log\det(SS+\epsilon I)$", grid_lim=grid_lim)
    _setup_map_axis(ax_itheta, r"$\log\det(I_{\theta,t}+\epsilon I)$", grid_lim=grid_lim)
    _draw_vector_field(
        ax_vf,
        dynamics_type=state.dynamics_type,
        a=state.a,
        b_param=state.b_param,
        grid_lim=grid_lim,
    )

    cbar_lambda = fig.colorbar(im_lambda, ax=ax_lambda, fraction=0.046, pad=0.02)
    cbar_lambda.set_label(r"mean($\lambda$)")
    cbar_iz = fig.colorbar(im_iz, ax=ax_iz, fraction=0.046, pad=0.02)
    cbar_iz.set_label(r"$\log\det(I_{z,t}+\epsilon I)$")
    cbar_ss = fig.colorbar(im_ss, ax=ax_ss, fraction=0.046, pad=0.02)
    cbar_ss.set_label(r"$\log\det(SS+\epsilon I)$")
    cbar_itheta = fig.colorbar(im_itheta, ax=ax_itheta, fraction=0.046, pad=0.02)
    cbar_itheta.set_label(r"$\log\det(I_{\theta,t}+\epsilon I)$")

    ax_ctrl.set_axis_off()
    ax_ctrl.set_title("Control Panel")

    ax_slider_a = ax_ctrl.inset_axes([0.08, 0.88, 0.84, 0.07])
    ax_text_a = ax_ctrl.inset_axes([0.08, 0.80, 0.36, 0.07])
    ax_slider_b = ax_ctrl.inset_axes([0.08, 0.70, 0.84, 0.07])
    ax_text_b = ax_ctrl.inset_axes([0.08, 0.62, 0.36, 0.07])
    ax_random = ax_ctrl.inset_axes([0.55, 0.62, 0.35, 0.10])
    ax_asym = ax_ctrl.inset_axes([0.08, 0.50, 0.40, 0.10])
    ax_dyn = ax_ctrl.inset_axes([0.08, 0.30, 0.50, 0.18])

    slider_a = Slider(
        ax_slider_a,
        str(slider_config["a_label"]),
        float(slider_config["a_min"]),
        float(slider_config["a_max"]),
        valinit=state.a,
    )
    text_a = TextBox(ax_text_a, str(slider_config["a_label"]), initial=f"{state.a:.4f}")

    slider_b = Slider(
        ax_slider_b,
        str(slider_config["b_label"]),
        float(slider_config["b_min"]),
        float(slider_config["b_max"]),
        valinit=state.b_param,
    )
    text_b = TextBox(ax_text_b, str(slider_config["b_label"]), initial=f"{state.b_param:.4f}")

    randomize_button = Button(ax_random, "randomize")
    asymmetric_toggle = CheckButtons(ax_asym, labels=["asymmetric"], actives=[state.asymmetric])
    dynamics_labels = [_DYNAMICS_LABELS[key] for key in DYNAMICS_CHOICES]
    dynamics_active = DYNAMICS_CHOICES.index(state.dynamics_type)
    dynamics_radio = RadioButtons(ax_dyn, labels=dynamics_labels, active=dynamics_active)
    diag_text = ax_ctrl.text(
        0.08,
        0.25,
        _format_diagnostics(state, bundle),
        transform=ax_ctrl.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        family="monospace",
    )

    sync_guard = {"active": False}

    def _set_textbox(textbox: TextBox, value: float) -> None:
        sync_guard["active"] = True
        textbox.set_val(f"{value:.4f}")
        sync_guard["active"] = False

    def _set_slider_bounds(slider: Slider, lower: float, upper: float, label: str) -> None:
        slider.valmin = float(lower)
        slider.valmax = float(upper)
        slider.ax.set_xlim(slider.valmin, slider.valmax)
        slider.label.set_text(label)

    def _apply_dynamics_preset(dynamics_type: str) -> None:
        preset = get_dynamics_preset(dynamics_type)
        _set_slider_bounds(
            slider_a,
            lower=float(preset["a_min"]),
            upper=float(preset["a_max"]),
            label=str(preset["a_label"]),
        )
        _set_slider_bounds(
            slider_b,
            lower=float(preset["b_min"]),
            upper=float(preset["b_max"]),
            label=str(preset["b_label"]),
        )
        text_a.label.set_text(str(preset["a_label"]))
        text_b.label.set_text(str(preset["b_label"]))
        _set_textbox(text_a, float(preset["a_init"]))
        _set_textbox(text_b, float(preset["b_init"]))
        slider_a.set_val(float(preset["a_init"]))
        slider_b.set_val(float(preset["b_init"]))

    def _refresh_maps() -> None:
        nonlocal bundle
        bundle = compute_map_bundle(state)
        _update_image_log(im_lambda, cbar_lambda, bundle.lambda_mean_map)
        _update_image(im_iz, cbar_iz, bundle.iz_logdet_map)
        _update_image(im_ss, cbar_ss, bundle.ss_logdet_map)
        _update_image(im_itheta, cbar_itheta, bundle.itheta_logdet_map)
        diag_text.set_text(_format_diagnostics(state, bundle))

    def _refresh_vector_field() -> None:
        _draw_vector_field(
            ax_vf,
            dynamics_type=state.dynamics_type,
            a=state.a,
            b_param=state.b_param,
            grid_lim=state.grid_lim,
        )
        diag_text.set_text(_format_diagnostics(state, bundle))

    def _refresh_all() -> None:
        _refresh_maps()
        _refresh_vector_field()

    def _on_slider_a(value: float) -> None:
        state.a = float(value)
        if not sync_guard["active"]:
            _set_textbox(text_a, state.a)
        _refresh_all()
        fig.canvas.draw_idle()

    def _on_slider_b(value: float) -> None:
        state.b_param = float(value)
        if not sync_guard["active"]:
            _set_textbox(text_b, state.b_param)
        _refresh_all()
        fig.canvas.draw_idle()

    def _on_text_a_submit(text: str) -> None:
        if sync_guard["active"]:
            return
        try:
            value = float(text)
        except ValueError:
            _set_textbox(text_a, float(slider_a.val))
            fig.canvas.draw_idle()
            return
        if value < slider_a.valmin or value > slider_a.valmax:
            _set_textbox(text_a, float(slider_a.val))
            fig.canvas.draw_idle()
            return
        slider_a.set_val(value)

    def _on_text_b_submit(text: str) -> None:
        if sync_guard["active"]:
            return
        try:
            value = float(text)
        except ValueError:
            _set_textbox(text_b, float(slider_b.val))
            fig.canvas.draw_idle()
            return
        if value < slider_b.valmin or value > slider_b.valmax:
            _set_textbox(text_b, float(slider_b.val))
            fig.canvas.draw_idle()
            return
        slider_b.set_val(value)

    def _on_randomize(_event: object) -> None:
        if state.rng is not None:
            seed = int(state.rng.integers(0, 2**31 - 1))
        else:
            seed = None
        state.c_raw = sample_raw_loading_matrix(obs_dim=state.obs_dim, dt=state.dt, seed=seed)
        state.c = apply_loading_asymmetry(state.c_raw, asymmetric=state.asymmetric)
        state.bias = compute_loading_bias(state.c, mean_firing=state.mean_firing)
        _refresh_maps()
        fig.canvas.draw_idle()

    def _on_asymmetric_toggle(_label: str) -> None:
        state.asymmetric = bool(asymmetric_toggle.get_status()[0])
        state.c = apply_loading_asymmetry(state.c_raw, asymmetric=state.asymmetric)
        state.bias = compute_loading_bias(state.c, mean_firing=state.mean_firing)
        _refresh_maps()
        fig.canvas.draw_idle()

    def _on_dynamics_select(label: str) -> None:
        state.dynamics_type = _LABEL_TO_DYNAMICS[str(label)]
        _apply_dynamics_preset(state.dynamics_type)
        fig.canvas.draw_idle()

    slider_a.on_changed(_on_slider_a)
    slider_b.on_changed(_on_slider_b)
    text_a.on_submit(_on_text_a_submit)
    text_b.on_submit(_on_text_b_submit)
    randomize_button.on_clicked(_on_randomize)
    asymmetric_toggle.on_clicked(_on_asymmetric_toggle)
    dynamics_radio.on_clicked(_on_dynamics_select)

    fig.suptitle("COSYNE Information Maps (Interactive)", fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.93, wspace=0.30, hspace=0.26)
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
