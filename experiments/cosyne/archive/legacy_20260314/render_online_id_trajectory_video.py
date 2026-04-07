from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from cosyne_common import (
    align_trace_length,
    frame_indices,
    load_json,
    load_summary_records,
    select_best_record,
)

from actdyn.utils.save_load import load_and_concatenate_rollouts, load_rollout
from mixed_family_lib import (
    CANONICAL_VECTORFIELD_STYLE,
    configure_runtime_device,
    load_model_bundle_checkpoint,
    prepare_selected_systems,
    true_dynamics_from_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a two-panel true vs inferred vectorfield trajectory video from a saved online-ID session."
    )
    parser.add_argument(
        "--summary",
        default="/home/hyungju/Desktop/active-dynamics/results/cosyne/metadynamics_online_id/summary.json",
        help="Summary file used to select a record when --record-path/--session-dir are omitted.",
    )
    parser.add_argument(
        "--record-path",
        default=None,
        help="Path to a saved online_id_record.json. Overrides summary-based selection.",
    )
    parser.add_argument(
        "--session-dir",
        default=None,
        help="Path to a saved online-ID session directory. Expects online_id_record.json inside.",
    )
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path for inferred vectorfield rendering.")
    parser.add_argument("--system-bank", default=None, help="Override system bank if not present in the record.")
    parser.add_argument("--system", default=None)
    parser.add_argument(
        "--policy",
        choices=["active_long", "active_short", "active_chunk", "async_windowed_update", "random", "no_policy"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--select-best-policy",
        choices=["active_long", "active_short", "active_chunk", "async_windowed_update", "random", "no_policy"],
        default="active_short",
        help="Policy used when auto-selecting a record from summary.json.",
    )
    parser.add_argument("--grid-n", type=int, default=25)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--video-filename", default="trajectory_vectorfields.mp4")
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def resolve_record(args: argparse.Namespace) -> dict:
    if args.record_path is not None:
        return load_json(Path(args.record_path))
    if args.session_dir is not None:
        return load_json(Path(args.session_dir) / "online_id_record.json")
    records = load_summary_records(args.summary)
    return select_best_record(
        records,
        filters={
            "system": args.system,
            "policy": args.policy,
            "seed": args.seed,
        },
        default_filter=("policy", args.select_best_policy),
    )


def resolve_rollout(record: dict):
    rollout_path = record.get("rollout_path")
    if rollout_path:
        path = Path(rollout_path)
        if path.exists():
            return load_rollout(str(path), device="cpu")
    session_dir = record.get("session_dir")
    if session_dir:
        rollouts_dir = Path(session_dir) / "rollouts"
        if rollouts_dir.exists():
            return load_and_concatenate_rollouts(str(rollouts_dir), device="cpu")
    raise FileNotFoundError("Could not resolve a saved rollout for the selected record.")


def resolve_checkpoint_path(record: dict, args: argparse.Namespace) -> str:
    checkpoint = args.checkpoint or record.get("checkpoint")
    if checkpoint is None:
        raise ValueError("No checkpoint available in the record. Pass --checkpoint explicitly.")
    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


def resolve_system_bank(record: dict, args: argparse.Namespace) -> str:
    system_bank = args.system_bank or record.get("system_bank")
    if system_bank is None:
        raise ValueError("No system bank available in the record. Pass --system-bank explicitly.")
    return str(system_bank)


def extract_rollout_arrays(rollout) -> tuple[np.ndarray, np.ndarray]:
    env_state = rollout["env_state"][0].detach().cpu().numpy()
    next_env_state = rollout["next_env_state"][0].detach().cpu().numpy()
    model_state = rollout["model_state"][0].detach().cpu().numpy()
    next_model_state = rollout["next_model_state"][0].detach().cpu().numpy()
    env_path = np.concatenate([env_state, next_env_state[-1:]], axis=0)
    model_path = np.concatenate([model_state, next_model_state[-1:]], axis=0)
    return env_path, model_path


def resolve_step_offset(record: dict, num_steps: int) -> int:
    total_steps = int(record.get("n_steps", num_steps))
    return max(0, total_steps - int(num_steps))


def align_embedding_trace(record: dict, num_steps: int, step_offset: int = 0) -> np.ndarray:
    values = np.asarray(record.get("embedding_trace", []), dtype=np.float32)
    if values.ndim == 1 and values.size > 0:
        values = values.reshape(1, -1)
    if values.size == 0:
        raise ValueError(
            "Selected record does not contain embedding_trace. "
            "Re-run online ID with the updated saver first."
        )
    if step_offset > 0 and values.shape[0] > step_offset:
        values = values[step_offset:]
    return align_trace_length(
        values,
        num_steps,
        empty_error=(
            "Selected record does not contain embedding_trace. "
            "Re-run online ID with the updated saver first."
        ),
    )


def compute_axis_limits(
    env_path: np.ndarray,
    model_path: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    del env_path, model_path
    return (-3.0, 3.0), (-3.0, 3.0)


def sanitize_field(field: np.ndarray, component_clip: float | None = None) -> np.ndarray:
    clean = np.nan_to_num(field.astype(np.float32, copy=True), nan=0.0, posinf=0.0, neginf=0.0)
    if component_clip is not None:
        clean = np.clip(clean, -float(component_clip), float(component_clip))
    return clean


def field_speed(field: np.ndarray) -> np.ndarray:
    return np.hypot(field[..., 0], field[..., 1])


def build_field_grid(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    grid_n: int,
    device: str | torch.device,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    x_np = np.linspace(xlim[0], xlim[1], int(grid_n), dtype=np.float32)
    y_np = np.linspace(ylim[0], ylim[1], int(grid_n), dtype=np.float32)
    X, Y = np.meshgrid(x_np, y_np, indexing="xy")
    z = torch.tensor(
        np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1),
        dtype=torch.float32,
        device=device,
    )
    return x_np, y_np, z


def resolve_spec_and_bundle(record: dict, args: argparse.Namespace):
    configure_runtime_device(seed=0)
    system_bank = resolve_system_bank(record, args)
    embedding_true = record.get("embedding_true", [])
    d_embed = max(len(embedding_true), 1)
    selected = prepare_selected_systems(
        system_bank=system_bank,
        systems=[record["system"]],
        embedding_mode=str(record.get("embedding_mode", "learned_system_id")),
        d_embed=d_embed,
    )
    checkpoint_path = resolve_checkpoint_path(record, args)
    bundle = load_model_bundle_checkpoint(checkpoint_path, selected)
    return selected[0], bundle, checkpoint_path


def precompute_vectorfields(
    *,
    spec,
    bundle,
    embedding_trace: np.ndarray,
    selected_indices: list[int],
    z_grid: torch.Tensor,
    grid_n: int,
    dynamics_scale: float,
) -> tuple[np.ndarray, dict[int, np.ndarray], float]:
    with torch.no_grad():
        true_field = (
            true_dynamics_from_spec(spec, z_grid, dynamics_scale=dynamics_scale)
            .detach()
            .cpu()
            .numpy()
            .reshape(grid_n, grid_n, 2)
        )
        true_field = sanitize_field(true_field)
        inferred_fields: dict[int, np.ndarray] = {}
        speed_candidates = [field_speed(true_field).reshape(-1)]
        for frame_idx in selected_indices:
            emb = embedding_trace[int(frame_idx)]
            e = torch.tensor(emb, dtype=torch.float32, device=z_grid.device).reshape(1, -1)
            e = e.repeat(z_grid.shape[0], 1)
            pred = (
                bundle.meta_dynamics(z_grid, e=e)
                .detach()
                .cpu()
                .numpy()
                .reshape(grid_n, grid_n, 2)
            )
            pred = sanitize_field(pred)
            inferred_fields[int(frame_idx)] = pred
            speed_candidates.append(field_speed(pred).reshape(-1))
    all_speeds = np.concatenate(speed_candidates, axis=0)
    finite_speeds = all_speeds[np.isfinite(all_speeds)]
    if finite_speeds.size == 0:
        max_speed = 1.0
    else:
        max_speed = float(np.max(finite_speeds))
    return true_field, inferred_fields, max(max_speed, 1e-6)


def resolve_meta_dynamics_device(bundle) -> torch.device:
    candidates = [
        getattr(bundle.meta_dynamics, "hypernet", None),
        getattr(bundle.meta_dynamics, "mean_dynamics", None),
        getattr(bundle, "hypernet", None),
        getattr(bundle, "mean_dynamics", None),
    ]
    for module in candidates:
        if module is None:
            continue
        if hasattr(module, "parameters"):
            try:
                return next(module.parameters()).device
            except StopIteration:
                continue
    return torch.device("cpu")


def make_frame_figure(
    *,
    record: dict,
    step_idx: int,
    actual_step: int,
    env_path: np.ndarray,
    model_path: np.ndarray,
    x_np: np.ndarray,
    y_np: np.ndarray,
    true_field: np.ndarray,
    inferred_field: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    speed_max: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    step = step_idx + 1
    env_hist = env_path[: step + 1]
    model_hist = model_path[: step + 1]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100, sharex=True, sharey=True)
    fig.subplots_adjust(bottom=0.16, top=0.90, wspace=0.08)

    norm = colors.Normalize(vmin=0.0, vmax=float(speed_max), clip=True)
    cmap = matplotlib.colormaps[str(CANONICAL_VECTORFIELD_STYLE["cmap"])]
    fields = (
        ("True vector field", true_field),
        ("Inferred vector field", inferred_field),
    )
    colorbar_mappable = None
    for ax, (title, field) in zip(axes, fields):
        U = field[..., 0]
        V = field[..., 1]
        speed = field_speed(field)
        stream = ax.streamplot(
            x_np,
            y_np,
            U,
            V,
            color=speed,
            cmap=cmap,
            norm=norm,
            density=float(CANONICAL_VECTORFIELD_STYLE["stream_density"]),
            linewidth=float(CANONICAL_VECTORFIELD_STYLE["line_width"]),
            arrowsize=float(CANONICAL_VECTORFIELD_STYLE["arrow_size"]),
        )
        stream.lines.set_cmap(cmap)
        stream.lines.set_norm(norm)
        if colorbar_mappable is None:
            colorbar_mappable = stream.lines
        ax.plot(env_hist[:, 0], env_hist[:, 1], color="white", lw=3.6, alpha=0.95)
        ax.plot(env_hist[:, 0], env_hist[:, 1], color="tab:blue", lw=2.0, label="true trajectory")
        ax.plot(model_hist[:, 0], model_hist[:, 1], color="white", lw=3.0, alpha=0.95)
        ax.plot(
            model_hist[:, 0],
            model_hist[:, 1],
            color="tab:orange",
            lw=2.0,
            ls="--",
            label="inferred trajectory",
        )
        ax.scatter(env_hist[0, 0], env_hist[0, 1], color="tab:blue", s=18, alpha=0.45)
        ax.scatter(model_hist[0, 0], model_hist[0, 1], color="tab:orange", s=18, alpha=0.45)
        ax.scatter(env_hist[-1, 0], env_hist[-1, 1], color="tab:blue", s=54, zorder=5)
        ax.scatter(model_hist[-1, 0], model_hist[-1, 1], color="tab:orange", marker="x", s=58, zorder=5)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].legend(loc="upper right", frameon=True)

    fig.suptitle(
        f"{record['system']} | {record['policy']} | seed {int(record['seed'])} | step {actual_step}",
        fontsize=14,
    )
    if colorbar_mappable is None:
        colorbar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar_mappable.set_array([])
    cbar = fig.colorbar(colorbar_mappable, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("Vector-field speed")
    return fig


def figure_to_frame(fig) -> np.ndarray:
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    return frame


def write_video(frames: list[np.ndarray], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing {len(frames)} frames -> {out_path}")
    iio.imwrite(
        out_path,
        np.stack(frames),
        fps=int(fps),
        codec="libx264",
        output_params=[
            "-pix_fmt",
            "yuv420p",
            "-profile:v",
            "high",
            "-crf",
            "12",
            "-movflags",
            "+faststart",
        ],
    )
    print("[INFO] Done.")


def render_video(record: dict, args: argparse.Namespace) -> Path:
    import matplotlib.pyplot as plt

    rollout = resolve_rollout(record)
    env_path, model_path = extract_rollout_arrays(rollout)
    num_steps = env_path.shape[0] - 1
    step_offset = resolve_step_offset(record, num_steps)
    embedding_trace = align_embedding_trace(record, num_steps, step_offset=step_offset)
    spec, bundle, checkpoint_path = resolve_spec_and_bundle(record, args)
    xlim, ylim = compute_axis_limits(env_path, model_path)
    field_device = resolve_meta_dynamics_device(bundle)
    x_np, y_np, z_grid = build_field_grid(xlim, ylim, args.grid_n, device=field_device)
    dynamics_scale = float(record.get("dynamics_scale", 1.0))
    frame_indices_list = frame_indices(num_steps, int(args.stride))
    true_field, inferred_fields, speed_max = precompute_vectorfields(
        spec=spec,
        bundle=bundle,
        embedding_trace=embedding_trace,
        selected_indices=frame_indices_list,
        z_grid=z_grid,
        grid_n=int(args.grid_n),
        dynamics_scale=dynamics_scale,
    )

    if args.output_path is not None:
        out_path = Path(args.output_path)
    else:
        session_dir = Path(record["session_dir"])
        out_path = session_dir / "video" / args.video_filename

    frames: list[np.ndarray] = []
    for frame_idx in frame_indices_list:
        actual_step = int(step_offset + frame_idx + 1)
        fig = make_frame_figure(
            record=record,
            step_idx=frame_idx,
            actual_step=actual_step,
            env_path=env_path,
            model_path=model_path,
            x_np=x_np,
            y_np=y_np,
            true_field=true_field,
            inferred_field=inferred_fields[int(frame_idx)],
            xlim=xlim,
            ylim=ylim,
            speed_max=speed_max,
        )
        frames.append(figure_to_frame(fig))
        plt.close(fig)
    write_video(frames, out_path, fps=int(args.fps))
    record["checkpoint"] = checkpoint_path
    return out_path


def main() -> None:
    args = parse_args()
    record = resolve_record(args)
    out_path = render_video(record, args)
    print(
        json.dumps(
            {
                "system": record["system"],
                "policy": record["policy"],
                "seed": int(record["seed"]),
                "rollout_path": record.get("rollout_path"),
                "record_path": record.get("record_path"),
                "checkpoint": record.get("checkpoint"),
                "video_path": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
