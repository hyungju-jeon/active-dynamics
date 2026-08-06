"""Utilities for analyzing generic training log chunks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from actdyn.utils.persistence import (
    concatenate_log_chunk as _concatenate_chunks,
    find_log_files as _find_log_files,
    load_log_file as _load_log_file,
)

MetricData = dict[str, list[Any]]
ModelData = dict[str, MetricData]
AllResults = dict[str, ModelData]

DEFAULT_METRIC_KEYWORDS = ("elbo", "loss", "objective", "train")


def _seed_sort_key(path: Path) -> tuple[int, str]:
    if path.name.startswith("seed_"):
        raw = path.name.split("seed_", 1)[1]
        if raw.isdigit():
            return (int(raw), path.name)
    return (10**9, path.name)


def _seed_value(seed_dir: Path) -> int | str:
    if seed_dir.name.startswith("seed_"):
        raw = seed_dir.name.split("seed_", 1)[1]
        if raw.isdigit():
            return int(raw)
    return seed_dir.name


def _log_group_key(log_file: Path) -> str:
    stem = log_file.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _safe_slug(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        return float(value)
    return None


def _numeric_columns(data: MetricData) -> list[str]:
    columns: list[str] = []
    for key, values in data.items():
        if not isinstance(values, list):
            continue
        for value in values:
            if _to_float(value) is not None:
                columns.append(key)
                break
    return columns


def _match_metric_columns(data: MetricData, keywords: tuple[str, ...]) -> list[str]:
    keywords_lc = tuple(k.lower() for k in keywords)
    return [
        col
        for col in _numeric_columns(data)
        if any(keyword in col.lower() for keyword in keywords_lc)
    ]


def _attach_seed_column(data: MetricData, seed: int | str) -> MetricData:
    if not data:
        return data

    lengths = [len(v) for v in data.values() if isinstance(v, list)]
    n_rows = max(lengths, default=0)
    if n_rows == 0:
        return data

    out = {k: list(v) if isinstance(v, list) else v for k, v in data.items()}
    if "seed" in out and isinstance(out["seed"], list):
        seed_values = list(out["seed"])[:n_rows]
        if len(seed_values) < n_rows:
            seed_values.extend([seed] * (n_rows - len(seed_values)))
        out["seed"] = seed_values
    else:
        out["seed"] = [seed] * n_rows
    return out


def load_seed_data(seed_dir: Path, is_offline: bool = False, verbose: bool = False) -> ModelData:
    """Load and concatenate log chunks for a single seed directory."""
    logs_dir = seed_dir / "logs"
    if not logs_dir.exists():
        if verbose:
            print(f"Skip {seed_dir.name}: missing logs directory")
        return {}

    patterns = ["offline_*.json"] if is_offline else ["log_*.json"]
    log_files = _find_log_files(logs_dir, patterns=patterns)
    if not log_files and not is_offline:
        # Fallback for runs that only saved generic json names.
        log_files = _find_log_files(logs_dir, patterns=["*.json"])

    if not log_files:
        if verbose:
            print(f"Skip {seed_dir.name}: no matching log files")
        return {}

    grouped: ModelData = {}
    for log_file in log_files:
        chunk = _load_log_file(log_file)
        if not chunk:
            continue

        group_key = _log_group_key(log_file)
        if group_key not in grouped:
            grouped[group_key] = chunk
            continue

        try:
            grouped[group_key] = _concatenate_chunks(grouped[group_key], chunk)
        except ValueError:
            # If schemas differ, keep a separate key instead of discarding data.
            grouped[log_file.stem] = chunk

    seed = _seed_value(seed_dir)
    for file_key in list(grouped.keys()):
        grouped[file_key] = _attach_seed_column(grouped[file_key], seed)
    return grouped


def load_model_data(model_dir: Path, is_offline: bool = False, verbose: bool = False) -> ModelData:
    """Load data for all seeds under one model directory."""
    seed_dirs = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")],
        key=_seed_sort_key,
    )
    if not seed_dirs:
        return {}

    model_data: ModelData = {}
    for seed_dir in seed_dirs:
        seed_data = load_seed_data(seed_dir, is_offline=is_offline, verbose=verbose)
        for file_key, chunk in seed_data.items():
            if file_key not in model_data:
                model_data[file_key] = chunk
                continue
            try:
                model_data[file_key] = _concatenate_chunks(model_data[file_key], chunk)
            except ValueError:
                common = set(model_data[file_key]).intersection(chunk)
                if not common:
                    continue
                left = {k: model_data[file_key][k] for k in common}
                right = {k: chunk[k] for k in common}
                model_data[file_key] = _concatenate_chunks(left, right)
    return model_data


def analyze_all_models(exp_folder: str, is_offline: bool = False) -> AllResults:
    """Load all model logs under an experiment result folder."""
    base_path = Path(exp_folder)
    if not base_path.exists():
        print(f"Error: missing experiment folder: {base_path}")
        return {}

    model_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    results: AllResults = {}
    for model_dir in model_dirs:
        model_data = load_model_data(model_dir, is_offline=is_offline)
        if model_data:
            results[model_dir.name] = model_data
    return results


def prepare_metric_plot_data(data: MetricData, metric_col: str) -> dict[str, Any] | None:
    """Compute aligned mean/std curves across seeds for one metric column."""
    if metric_col not in data or "seed" not in data:
        return None

    values = data.get(metric_col, [])
    seeds = data.get("seed", [])
    if not isinstance(values, list) or not isinstance(seeds, list):
        return None
    if not values or not seeds:
        return None

    n_rows = min(len(values), len(seeds))
    step_values = data.get("step", None)
    has_steps = isinstance(step_values, list) and len(step_values) >= n_rows

    per_seed: dict[int | str, dict[str, list[Any]]] = {}
    for idx in range(n_rows):
        seed = seeds[idx]
        value = _to_float(values[idx])
        if value is None:
            continue

        if seed not in per_seed:
            per_seed[seed] = {"values": [], "steps": []}
        per_seed[seed]["values"].append(value)
        if has_steps:
            per_seed[seed]["steps"].append(step_values[idx])

    if not per_seed:
        return None

    ordered_seeds = sorted(per_seed.keys(), key=lambda s: str(s))
    if has_steps:
        all_steps: set[float] = set()
        step_maps: list[dict[float, float]] = []
        for seed in ordered_seeds:
            pairs = zip(per_seed[seed]["steps"], per_seed[seed]["values"])
            step_map: dict[float, float] = {}
            for step, value in pairs:
                step_f = _to_float(step)
                if step_f is None:
                    continue
                step_map[step_f] = value
            if step_map:
                all_steps.update(step_map.keys())
            step_maps.append(step_map)

        if not all_steps:
            return None

        ordered_steps = np.array(sorted(all_steps), dtype=np.float64)
        matrix = np.full((len(ordered_seeds), len(ordered_steps)), np.nan, dtype=np.float64)
        step_index = {step: i for i, step in enumerate(ordered_steps)}
        for row_idx, seed_map in enumerate(step_maps):
            for step, value in seed_map.items():
                matrix[row_idx, step_index[step]] = value

        mean_values = np.nanmean(matrix, axis=0)
        std_values = np.nanstd(matrix, axis=0)
        valid_mask = ~np.isnan(mean_values)
        if not np.any(valid_mask):
            return None

        return {
            "time_steps": ordered_steps[valid_mask],
            "mean": mean_values[valid_mask],
            "std": std_values[valid_mask],
            "n_seeds": len(ordered_seeds),
        }

    min_len = min(len(per_seed[seed]["values"]) for seed in ordered_seeds)
    if min_len == 0:
        return None

    matrix = np.stack(
        [np.array(per_seed[seed]["values"][:min_len], dtype=np.float64) for seed in ordered_seeds]
    )
    return {
        "time_steps": np.arange(min_len),
        "mean": np.mean(matrix, axis=0),
        "std": np.std(matrix, axis=0),
        "n_seeds": len(ordered_seeds),
    }


def plot_metric_curve(ax, plot_data: dict[str, Any], metric_name: str, model_name: str) -> None:
    """Plot mean/std metric curve for one model."""
    time_steps = plot_data["time_steps"]
    mean_values = plot_data["mean"]
    std_values = plot_data["std"]
    n_seeds = plot_data["n_seeds"]

    ax.plot(time_steps, mean_values, linewidth=2, label=f"Mean (n={n_seeds})")
    ax.fill_between(time_steps, mean_values - std_values, mean_values + std_values, alpha=0.25)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Over Time - {model_name}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metrics_over_time(
    results: AllResults,
    output_dir: Optional[str] = None,
    keywords: tuple[str, ...] = DEFAULT_METRIC_KEYWORDS,
) -> list[str]:
    """Generate metric-over-time plots for each model/log group."""
    import matplotlib.pyplot as plt

    output_path = Path(output_dir or ".")
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for model_name, model_data in results.items():
        for file_key, data in model_data.items():
            metric_columns = _match_metric_columns(data, keywords=keywords)
            for metric_col in metric_columns:
                plot_data = prepare_metric_plot_data(data, metric_col)
                if plot_data is None:
                    continue

                _, ax = plt.subplots(figsize=(10, 6))
                plot_metric_curve(ax, plot_data, metric_col, model_name)

                filename = (
                    f"metric_plot_{_safe_slug(model_name)}_{_safe_slug(file_key)}_"
                    f"{_safe_slug(metric_col)}.png"
                )
                plot_path = output_path / filename
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                saved_paths.append(str(plot_path))
    return saved_paths


def plot_all_models_metric_comparison(
    results: AllResults,
    output_dir: Optional[str] = None,
    keywords: tuple[str, ...] = DEFAULT_METRIC_KEYWORDS,
) -> list[str]:
    """Generate comparison plots with all models for the same metric/log group."""
    import matplotlib.pyplot as plt

    output_path = Path(output_dir or ".")
    output_path.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, model_data in results.items():
        for file_key, data in model_data.items():
            for metric_col in _match_metric_columns(data, keywords=keywords):
                plot_data = prepare_metric_plot_data(data, metric_col)
                if plot_data is None:
                    continue
                key = f"{file_key}::{metric_col}"
                grouped.setdefault(key, {})[model_name] = plot_data

    saved_paths: list[str] = []
    for key, by_model in grouped.items():
        if len(by_model) < 2:
            continue

        file_key, metric_col = key.split("::", 1)
        _, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(by_model)))

        for idx, (model_name, plot_data) in enumerate(sorted(by_model.items())):
            time_steps = plot_data["time_steps"]
            mean_values = plot_data["mean"]
            std_values = plot_data["std"]
            n_seeds = plot_data["n_seeds"]
            color = colors[idx]

            ax.plot(
                time_steps,
                mean_values,
                linewidth=2,
                color=color,
                label=f"{model_name} (n={n_seeds})",
            )
            ax.fill_between(
                time_steps,
                mean_values - std_values,
                mean_values + std_values,
                alpha=0.2,
                color=color,
            )

        ax.set_xlabel("Training Steps")
        ax.set_ylabel(metric_col)
        ax.set_title(f"{metric_col} Comparison - {file_key}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        filename = f"metric_comparison_{_safe_slug(file_key)}_{_safe_slug(metric_col)}.png"
        plot_path = output_path / filename
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(str(plot_path))
    return saved_paths


def summarize_results(results: AllResults) -> dict[str, dict[str, dict[str, float]]]:
    """Compute final-step summary stats for each model/log group/metric."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_data in results.items():
        summary[model_name] = {}
        for file_key, data in model_data.items():
            file_stats: dict[str, float] = {}
            for metric_col in _numeric_columns(data):
                if metric_col in {"seed", "step"}:
                    continue
                plot_data = prepare_metric_plot_data(data, metric_col)
                if plot_data is None:
                    continue
                file_stats[f"{metric_col}_last_mean"] = float(plot_data["mean"][-1])
                file_stats[f"{metric_col}_last_std"] = float(plot_data["std"][-1])
                file_stats[f"{metric_col}_seed_count"] = float(plot_data["n_seeds"])
            summary[model_name][file_key] = file_stats
    return summary


def save_summary_results(
    results: AllResults, exp_folder: str, output_file: Optional[str] = None
) -> dict[str, dict[str, dict[str, float]]]:
    """Save summary statistics to a JSON file and return them."""
    summary = summarize_results(results)
    output_path = Path(output_file) if output_file else Path(exp_folder) / "analysis_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def print_summary(results: AllResults) -> None:
    """Print compact final-step summary statistics."""
    summary = summarize_results(results)
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    for model_name, model_summary in summary.items():
        print(f"\nModel: {model_name}")
        for file_key, stats in model_summary.items():
            print(f"  Log group: {file_key}")
            if not stats:
                print("    No numeric metrics")
                continue

            metric_names = sorted(
                set(
                    key[: -len("_last_mean")]
                    for key in stats
                    if key.endswith("_last_mean")
                )
            )
            for metric in metric_names:
                mean = stats.get(f"{metric}_last_mean")
                std = stats.get(f"{metric}_last_std")
                n = int(stats.get(f"{metric}_seed_count", 0.0))
                if mean is None or std is None:
                    continue
                print(f"    {metric}: {mean:.4f} ± {std:.4f} (n={n})")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze training result logs.")
    parser.add_argument("exp_folder", type=str, help="Path to training result root directory")
    parser.add_argument("--offline", action="store_true", help="Use offline_* logs")
    parser.add_argument("--summary", action="store_true", help="Print summary to stdout")
    parser.add_argument("--plot", action="store_true", help="Generate per-model metric plots")
    parser.add_argument("--compare", action="store_true", help="Generate metric comparison plots across models")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument(
        "--save-summary",
        action="store_true",
        help="Save summary JSON to <exp_folder>/analysis_summary.json",
    )
    args = parser.parse_args()

    results = analyze_all_models(args.exp_folder, is_offline=args.offline)
    if not results:
        print("No results found.")
        return 1

    if args.summary:
        print_summary(results)
    if args.save_summary:
        save_summary_results(results, exp_folder=args.exp_folder)
    if args.plot:
        paths = plot_metrics_over_time(results, output_dir=args.output_dir)
        print(f"Saved {len(paths)} plot(s).")
    if args.compare:
        paths = plot_all_models_metric_comparison(results, output_dir=args.output_dir)
        print(f"Saved {len(paths)} comparison plot(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
