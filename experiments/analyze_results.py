# %%
from operator import is_
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from actdyn.utils.save_load import load_and_concatenate_logs, find_log_files


def load_log_file(file_path: Path) -> Dict[str, List[Any]]:
    """Load a single log file and return as dictionary of lists."""
    if file_path.suffix.lower() == ".json":
        return load_log_file(file_path)
    else:
        print(f"Warning: Unsupported file format {file_path.suffix}")
        return {}


def load_seed_data(seed_dir: Path, is_offline: bool = False, verbose=False) -> Dict[str, List[Any]]:
    """
    Load and concatenate sequential log files for a single seed.
    """
    if verbose:
        print(f"  Processing seed: {seed_dir.name}")

    # Check if logs directory exists
    logs_dir = seed_dir / "logs"
    if not logs_dir.exists():
        print(f"Warning: logs directory not found in {seed_dir}")
        return {}

    # Use the refactored function from helpers
    if is_offline:
        pattern = "offline_*.json"
        concatenated_data = load_and_concatenate_logs(logs_dir, pattern)
    else:
        concatenated_data = load_and_concatenate_logs(logs_dir)

    return concatenated_data


def load_model_data(
    model_dir: Path, is_offline: bool = False, verbose=False
) -> Dict[str, Dict[str, List[Any]]]:
    """Load and concatenate data for all seeds of a model."""
    if verbose:
        print(f"\nProcessing model: {model_dir.name}")
    # Find all seed directories
    seed_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]

    if not seed_dirs:
        print(f"  No seed directories found in {model_dir}")
        return {}

    if verbose:
        print(f"  Found {len(seed_dirs)} seed directories")

    # Dictionary to store concatenated data for each log file type
    model_data = {}
    # Load data from each seed
    for seed_dir in sorted(seed_dirs):
        seed_data = load_seed_data(seed_dir, is_offline=is_offline)

        # Concatenate with existing data
        for file_key, data in seed_data.items():
            if file_key not in model_data:
                model_data[file_key] = []
            model_data[file_key].append(data)

    return model_data


def analyze_all_models(exp_folder: str, is_offline: bool = False) -> Dict[str, Dict[str, Any]]:
    """Analyze all models in the experiment folder."""
    base_path = Path(exp_folder)
    if not base_path.exists():
        print(f"Error: Base folder does not exist: {exp_folder}")
        return {}

    print(f"Analyzing results in: {exp_folder}")
    # Find all model directories
    model_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not model_dirs:
        print("No model directories found!")
        return {}
    print(f"Found {len(model_dirs)} model directories")

    results = {}
    for model_dir in sorted(model_dirs):
        model_name = model_dir.name
        # Load concatenated data for this model
        model_data = load_model_data(model_dir, is_offline=is_offline)
        if not model_data:
            print(f"  No data loaded for {model_name}")
            continue
        results[model_name] = model_data

    return results


def save_summary_results(
    results: Dict[str, Dict[str, Any]], exp_folder: str, output_file: Optional[str] = None
) -> Dict[str, List[Any]]:
    """Save summary statistics to a JSON file."""
    if output_file is None:
        output_file = os.path.join(exp_folder, "analysis_summary.json")

    return save_analysis_summary(results, output_file)


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)

        stats = model_results["statistics"]
        if not stats:
            print("  No statistics computed")
            continue

        for file_key, file_stats in stats.items():
            print(f"\n  Log file: {file_key}")

            # Group metrics by base name
            metrics = {}
            for key, value in file_stats.items():
                if key.endswith("_mean"):
                    metric_name = key[:-5]  # Remove '_mean'
                    if metric_name not in metrics:
                        metrics[metric_name] = {}
                    metrics[metric_name]["mean"] = value
                elif key.endswith("_std"):
                    metric_name = key[:-4]  # Remove '_std'
                    if metric_name not in metrics:
                        metrics[metric_name] = {}
                    metrics[metric_name]["std"] = value
                elif key.endswith("_count_seeds"):
                    metric_name = key[:-12]  # Remove '_count_seeds'
                    if metric_name not in metrics:
                        metrics[metric_name] = {}
                    metrics[metric_name]["seeds"] = value

            # Print metrics in a nice format
            for metric_name, values in metrics.items():
                mean_val = values.get("mean", "N/A")
                std_val = values.get("std", "N/A")
                seeds = values.get("seeds", "N/A")

                if isinstance(mean_val, (int, float, np.number)) and isinstance(
                    std_val, (int, float, np.number)
                ):
                    if not (np.isnan(mean_val) or np.isnan(std_val)):
                        print(
                            f"    {metric_name}: {mean_val:.4f} ± {std_val:.4f} (n={seeds} seeds)"
                        )
                    else:
                        print(f"    {metric_name}: N/A ± N/A (n={seeds} seeds)")
                else:
                    print(f"    {metric_name}: {mean_val} ± {std_val} (n={seeds} seeds)")


def plot_elbo_over_time(results: Dict[str, Dict[str, Any]], output_dir: Optional[str] = None):
    """Plot ELBO over time with mean and shaded standard deviation area."""
    # Note: exp_folder would need to be passed as parameter if output_dir is None

    # Look for ELBO data in results
    elbo_data = {}

    for model_name, model_results in results.items():
        data = model_results["data"]

        # Look for files that might contain ELBO data
        elbo_file_keys = []
        for file_key in data.keys():
            # Check if any columns contain 'elbo', 'loss', or 'objective'
            file_data = data[file_key]
            elbo_columns = []

            for col_name in file_data.keys():
                if any(
                    keyword in col_name.lower()
                    for keyword in ["elbo", "loss", "objective", "train"]
                ):
                    # Check if it's numeric
                    if col_name in get_numeric_columns(file_data):
                        elbo_columns.append(col_name)

            if elbo_columns:
                elbo_file_keys.append((file_key, elbo_columns))

        if elbo_file_keys:
            elbo_data[model_name] = elbo_file_keys

    if not elbo_data:
        print("No ELBO/loss data found for plotting")
        return

    print(f"\nGenerating ELBO plots...")

    # Create plots for each model and each ELBO column
    for model_name, file_data in elbo_data.items():
        print(f"  Plotting ELBO for model: {model_name}")

        for file_key, elbo_columns in file_data:
            model_data = results[model_name]["data"][file_key]

            for elbo_col in elbo_columns:
                print(f"    Plotting {elbo_col} from {file_key}")

                # Add debug info about the data
                print(f"        Data shape: {len(model_data[elbo_col])} records")
                if "step" in model_data:
                    print(
                        f"        Step range: {min(model_data['step'])}-{max(model_data['step'])}"
                    )

                # Prepare data for plotting
                plot_data = prepare_elbo_plot_data(model_data, elbo_col)

                if plot_data:
                    print(
                        f"        Plot data prepared: {len(plot_data['time_steps'])} points, {plot_data['n_seeds']} seeds"
                    )

                    # Create the plot
                    _, ax = plt.subplots(figsize=(10, 6))

                    # Plot mean line and shaded std area
                    plot_elbo_curve(ax, plot_data, elbo_col, model_name)

                    # Save plot
                    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
                    safe_file_key = file_key.replace("/", "_").replace(" ", "_")
                    safe_elbo_col = elbo_col.replace("/", "_").replace(" ", "_")

                    plot_filename = (
                        f"elbo_plot_{safe_model_name}_{safe_file_key}_{safe_elbo_col}.png"
                    )
                    if output_dir:
                        plot_path = os.path.join(output_dir, plot_filename)
                    else:
                        plot_path = plot_filename

                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    print(f"      Saved plot: {plot_filename}")
                else:
                    print(
                        f"      No valid data for {elbo_col} (prepare_elbo_plot_data returned None)"
                    )


def prepare_elbo_plot_data(data: Dict[str, List[Any]], elbo_col: str) -> Optional[Dict[str, Any]]:
    """Prepare ELBO data for plotting by computing mean and std across seeds."""
    if elbo_col not in data or "seed" not in data:
        return None

    # Check if we have step information (from sequential logs)
    has_steps = "step" in data

    # Group data by seed
    seed_data = {}
    for i, seed in enumerate(data["seed"]):
        if seed not in seed_data:
            seed_data[seed] = {"values": [], "steps": []}

        if i < len(data[elbo_col]) and data[elbo_col][i] is not None:
            seed_data[seed]["values"].append(data[elbo_col][i])
            if has_steps and i < len(data["step"]):
                seed_data[seed]["steps"].append(data["step"][i])
            else:
                seed_data[seed]["steps"].append(i)

    if not seed_data:
        return None

    # If we have steps, use them for x-axis; otherwise use indices
    if has_steps:
        # Find common step range across all seeds
        all_steps = set()
        for seed_values in seed_data.values():
            all_steps.update(seed_values["steps"])

        common_steps = sorted(all_steps)

        if not common_steps:
            return None

        # Create aligned data for each seed
        seed_arrays = []
        valid_steps = []

        for step in common_steps:
            step_values = []
            for seed, seed_values in seed_data.items():
                if step in seed_values["steps"]:
                    step_idx = seed_values["steps"].index(step)
                    step_values.append(seed_values["values"][step_idx])
                else:
                    step_values.append(np.nan)

            # Only keep steps where we have data from at least one seed
            if not all(np.isnan(v) for v in step_values):
                if not seed_arrays:
                    seed_arrays = [[] for _ in range(len(seed_data))]

                for i, val in enumerate(step_values):
                    seed_arrays[i].append(val)
                valid_steps.append(step)

        if not seed_arrays or not valid_steps:
            return None

        # Convert to numpy arrays
        seed_arrays = [np.array(arr) for arr in seed_arrays]
        time_steps = np.array(valid_steps)

    else:
        # Use indices - find minimum length across all seeds
        min_length = min(len(seed_values["values"]) for seed_values in seed_data.values())

        if min_length == 0:
            return None

        # Create arrays for plotting
        time_steps = np.arange(min_length)
        seed_arrays = []

        for seed, seed_values in seed_data.items():
            # Take only the first min_length values to ensure all seeds have same length
            seed_array = np.array(seed_values["values"][:min_length])
            seed_arrays.append(seed_array)

    # Handle case where we have varying lengths or NaN values
    if has_steps:
        # For step-based data, compute statistics ignoring NaN values
        all_seeds_matrix = np.array(seed_arrays)  # Shape: (n_seeds, n_timesteps)

        mean_values = np.nanmean(all_seeds_matrix, axis=0)
        std_values = np.nanstd(all_seeds_matrix, axis=0)

        # Remove time points where all seeds are NaN
        valid_mask = ~np.isnan(mean_values)
        if not np.any(valid_mask):
            return None

        time_steps = time_steps[valid_mask]
        mean_values = mean_values[valid_mask]
        std_values = std_values[valid_mask]

    else:
        # For index-based data, all should have same length
        all_seeds_data = np.stack(seed_arrays)  # Shape: (n_seeds, n_timesteps)
        mean_values = np.mean(all_seeds_data, axis=0)
        std_values = np.std(all_seeds_data, axis=0)

    return {
        "time_steps": time_steps,
        "mean": mean_values,
        "std": std_values,
        "n_seeds": len(seed_arrays),
    }


def plot_elbo_curve(ax, plot_data: Dict[str, Any], elbo_col: str, model_name: str):
    """Plot ELBO curve with mean line and shaded standard deviation area."""
    time_steps = plot_data["time_steps"]
    mean_values = plot_data["mean"]
    std_values = plot_data["std"]
    n_seeds = plot_data["n_seeds"]

    # Plot mean line
    ax.plot(time_steps, mean_values, linewidth=2, label=f"Mean (n={n_seeds} seeds)")

    # Plot shaded standard deviation area
    ax.fill_between(
        time_steps, mean_values - std_values, mean_values + std_values, alpha=0.3, label="±1 std"
    )

    # Formatting
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(elbo_col)
    ax.set_title(f"{elbo_col} Over Time - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add some styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add some debug info
    print(
        f"        Plot range: steps {time_steps[0]:.0f}-{time_steps[-1]:.0f}, "
        f"values {mean_values.min():.4f}-{mean_values.max():.4f}"
    )


def plot_all_models_elbo_comparison(
    results: Dict[str, Dict[str, Any]], output_dir: Optional[str] = None
):
    """Create comparison plots with all models on the same plot."""
    if output_dir is None:
        output_dir = exp_folder

    # Collect ELBO data from all models
    all_elbo_data = {}

    for model_name, model_results in results.items():
        data = model_results["data"]

        for file_key, file_data in data.items():
            elbo_columns = []
            for col_name in file_data.keys():
                if any(
                    keyword in col_name.lower()
                    for keyword in ["elbo", "loss", "objective", "train"]
                ):
                    if col_name in get_numeric_columns(file_data):
                        elbo_columns.append(col_name)

            for elbo_col in elbo_columns:
                plot_data = prepare_elbo_plot_data(file_data, elbo_col)
                if plot_data:
                    key = f"{file_key}_{elbo_col}"
                    if key not in all_elbo_data:
                        all_elbo_data[key] = {}
                    all_elbo_data[key][model_name] = plot_data

    # Create comparison plots
    for data_key, model_data in all_elbo_data.items():
        if len(model_data) > 1:  # Only create comparison if we have multiple models
            _, ax = plt.subplots(figsize=(12, 8))

            colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(model_data)))

            for i, (model_name, plot_data) in enumerate(model_data.items()):
                time_steps = plot_data["time_steps"]
                mean_values = plot_data["mean"]
                std_values = plot_data["std"]
                n_seeds = plot_data["n_seeds"]

                color = colors[i]

                # Plot mean line
                ax.plot(
                    time_steps,
                    mean_values,
                    linewidth=2,
                    color=color,
                    label=f"{model_name} (n={n_seeds})",
                )

                # Plot shaded std area
                ax.fill_between(
                    time_steps,
                    mean_values - std_values,
                    mean_values + std_values,
                    alpha=0.2,
                    color=color,
                )

            # Formatting
            file_key, elbo_col = data_key.rsplit("_", 1)
            ax.set_xlabel("Time Steps")
            ax.set_ylabel(elbo_col)
            ax.set_title(f"{elbo_col} Comparison - All Models ({file_key})")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Save comparison plot
            safe_data_key = data_key.replace("/", "_").replace(" ", "_")
            plot_filename = f"elbo_comparison_{safe_data_key}.png"
            plot_path = os.path.join(output_dir, plot_filename)

            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  Saved comparison plot: {plot_filename}")


if __name__ == "__main__":
    """Main function to run the analysis."""
    print("Starting hierarchical results analysis...")
    exp_folder = Path("/home/hyungju/Desktop/active-dynamics/results/offline_debug/sweep")

    # Analyze all models
    results = analyze_all_models(exp_folder)

    # if not results:
    #     print("No results to analyze!")
    #     return {}, {}

    # # Print summary
    # print_summary(results)

    # # Save summary to CSV
    # summary_data = save_summary_results(results)

    # # Generate ELBO plots
    # plot_elbo_over_time(results)
    # plot_all_models_elbo_comparison(results)

    # print(f"\nAnalysis complete!")
    # print(f"Processed {len(results)} models")
