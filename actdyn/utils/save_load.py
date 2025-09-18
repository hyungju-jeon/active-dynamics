import os
import json
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path

from actdyn.utils.rollout import Rollout, RolloutBuffer


def save_logger(logger, path="logs/log.json"):
    logger.append_to_json(path)


def save_model(model, path="checkpoints/model"):
    os.makedirs(path, exist_ok=True)
    for i, m in enumerate(model.models):
        with open(f"{path}/model_{i}.pkl", "wb") as f:
            pickle.dump(m, f)


def load_model(model, path="checkpoints/model"):
    model.models.clear()
    for i in range(len(os.listdir(path))):
        with open(f"{path}/model_{i}.pkl", "rb") as f:
            model.models.append(pickle.load(f))


def save_rollout(rollout, path="checkpoints/rollout.pkl"):
    """Save rollout buffer to disk."""
    if isinstance(rollout, Rollout):
        rollout_copy = rollout.copy()  # Ensure we don't modify the original
        rollout_copy.finalize()  # Finalize the rollout before saving
    elif isinstance(rollout, RolloutBuffer):
        rollout_copy = rollout.copy()  # Ensure we don't modify the originalg
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(rollout_copy, f)


def load_rollout(path="checkpoints/rollout.pkl") -> Rollout:
    """Load rollout buffer from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_and_concatenate_rollouts(
    file_path: str, pattern: str = "rollout_*.pkl", device="cpu"
) -> Rollout:
    """Load and concatenate rollout buffers from disk."""
    if not os.path.isdir(file_path):
        raise FileNotFoundError(
            f"Rollouts directory not found: {file_path}. Generate rollouts first."
        )

    rollout_files = [f for f in os.listdir(file_path) if f.endswith(".pkl")]
    if not rollout_files:
        raise FileNotFoundError(f"No .pkl rollouts found under {file_path}")
    try:
        rollout_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    except Exception:
        rollout_files.sort()

    rollout = Rollout(device=device)
    for rf in rollout_files:
        rp = os.path.join(file_path, rf)
        _r = load_rollout(rp)
        rollout.add(**_r.to(device)._data)

    rollout.finalize()

    return rollout


def load_and_concatenate_logs(file_path: str, pattern: str = "log_*.json") -> Dict[str, List[Any]]:
    """Load and concatenate log files"""
    log_files = find_log_files(Path(file_path), patterns=[pattern])
    if not log_files:
        print(f"No log files found matching pattern '{pattern}' in {file_path}.")
        return {}
    concatenated_data = {}
    for log_file in log_files:
        log_data = load_log_file(log_file)
        concatenated_data = concatenate_log_chunk(concatenated_data, log_data)
    if pattern.startswith("offline_"):
        for key, value in concatenated_data.items():
            concatenated_data[key] = value[0]

    return concatenated_data


def load_log_file(file_path: Path) -> Dict[str, List[Any]]:
    """Load a JSON file and return as dictionary of lists."""
    try:
        with open(file_path, "r") as f:
            json_data = json.load(f)

        # Convert to consistent format (dict of lists)
        if isinstance(json_data, list):
            # List of dictionaries
            if json_data and isinstance(json_data[0], dict):
                data = {}
                for record in json_data:
                    for key, value in record.items():
                        if key not in data:
                            data[key] = []
                        data[key].append(value)
                return data
            else:
                # List of values
                return {"values": json_data}

        elif isinstance(json_data, dict):
            # Check if it's already in the right format (dict of lists)
            if all(isinstance(v, list) for v in json_data.values()):
                return json_data
            else:
                # Single record, convert to lists
                return {k: [v] for k, v in json_data.items()}

        else:
            # Single value
            return {"value": [json_data]}

    except Exception as e:
        print(f"Warning: Could not load JSON {file_path}: {e}")
        return {}


def find_log_files(logs_dir: Path, patterns: Optional[List[str]] = None) -> List[Path]:
    """Find all log files in a logs directory."""
    if patterns is None:
        patterns = ["log_*.json", "*.json"]

    log_files = []
    for pattern in patterns:
        log_files.extend(logs_dir.glob(pattern))

    def sort_key(p: Path):
        import re

        parts = re.split(r"(\d+)", p.name)
        return [int(part) if part.isdigit() else part for part in parts]

    return sorted(log_files, key=sort_key)


def concatenate_log_chunk(
    concatenated_data: Dict[str, List[Any]], new_data: Dict[str, List[Any]]
) -> Dict[str, List[Any]]:
    """Concatenate multiple data dictionaries."""
    # If concatenated_data is empty, initialize it with new_data
    if not concatenated_data:
        return new_data

    # Find all unique keys, throw error if keys are not the same
    if set(concatenated_data.keys()) != set(new_data.keys()):
        raise ValueError(
            "Keys in concatenated_data and new_data do not match. "
            f"concatenated_data keys: {set(concatenated_data.keys())}, "
            f"new_data keys: {set(new_data.keys())}"
        )

    # Concatenate data for each key
    concatenated = {}
    for key in concatenated_data.keys():
        concatenated[key] = concatenated_data[key] + new_data[key]

    return concatenated


# def get_numeric_columns(
#     data: Dict[str, List[Any]], exclude_cols: Optional[List[str]] = None
# ) -> List[str]:
#     """Identify numeric columns in the data."""
#     if exclude_cols is None:
#         exclude_cols = ["seed", "model"]

#     numeric_cols = []
#     for key, values in data.items():
#         if key in exclude_cols:
#             continue

#         # Check if all non-None values are numeric
#         numeric_values = [v for v in values if v is not None]
#         if numeric_values and all(isinstance(v, (int, float, np.number)) for v in numeric_values):
#             numeric_cols.append(key)

#     return numeric_cols


# def compute_statistics_by_seed(
#     data: Dict[str, List[Any]], numeric_cols: List[str]
# ) -> Dict[str, Any]:
#     """Compute statistics grouped by seed first, then across seeds."""
#     if not numeric_cols or "seed" not in data:
#         return {}

#     # Group data by seed
#     seed_data = {}
#     for i, seed in enumerate(data["seed"]):
#         if seed not in seed_data:
#             seed_data[seed] = {col: [] for col in numeric_cols}

#         for col in numeric_cols:
#             if i < len(data[col]) and data[col][i] is not None:
#                 seed_data[seed][col].append(data[col][i])

#     # Compute mean for each seed and column
#     seed_means = {}
#     for seed, seed_values in seed_data.items():
#         seed_means[seed] = {}
#         for col in numeric_cols:
#             if seed_values[col]:
#                 seed_means[seed][col] = np.mean(seed_values[col])
#             else:
#                 seed_means[seed][col] = np.nan

#     # Compute statistics across seeds
#     stats = {}
#     for col in numeric_cols:
#         values = [
#             seed_means[seed][col] for seed in seed_means if not np.isnan(seed_means[seed][col])
#         ]

#         if values:
#             stats[f"{col}_mean"] = np.mean(values)
#             stats[f"{col}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
#             stats[f"{col}_min"] = np.min(values)
#             stats[f"{col}_max"] = np.max(values)
#             stats[f"{col}_count_seeds"] = len(values)
#         else:
#             stats[f"{col}_mean"] = np.nan
#             stats[f"{col}_std"] = np.nan
#             stats[f"{col}_min"] = np.nan
#             stats[f"{col}_max"] = np.nan
#             stats[f"{col}_count_seeds"] = 0

#     return stats


# def compute_statistics_direct(
#     data: Dict[str, List[Any]], numeric_cols: List[str]
# ) -> Dict[str, Any]:
#     """Compute statistics directly on all data."""
#     stats = {}

#     for col in numeric_cols:
#         values = [v for v in data[col] if v is not None and not np.isnan(v)]

#         if values:
#             stats[f"{col}_mean"] = np.mean(values)
#             stats[f"{col}_std"] = np.std(values, ddof=1) if len(values) > 1 else 0.0
#             stats[f"{col}_min"] = np.min(values)
#             stats[f"{col}_max"] = np.max(values)
#             stats[f"{col}_count"] = len(values)
#         else:
#             stats[f"{col}_mean"] = np.nan
#             stats[f"{col}_std"] = np.nan
#             stats[f"{col}_min"] = np.nan
#             stats[f"{col}_max"] = np.nan
#             stats[f"{col}_count"] = 0

#     return stats


# def compute_statistics(data: Dict[str, List[Any]], group_by_seed: bool = True) -> Dict[str, Any]:
#     """Compute mean and std statistics for numeric columns."""
#     numeric_cols = get_numeric_columns(data)

#     if not numeric_cols:
#         return {}

#     if group_by_seed and "seed" in data:
#         return compute_statistics_by_seed(data, numeric_cols)
#     else:
#         return compute_statistics_direct(data, numeric_cols)


# def save_analysis_summary(
#     results: Dict[str, Dict[str, Any]], output_file: str
# ) -> Dict[str, List[Any]]:
#     """Save summary statistics to a JSON file."""
#     # Prepare data for JSON
#     summary_data = []

#     # Fill data
#     for model_name, model_results in results.items():
#         stats = model_results["statistics"]

#         for file_key, file_stats in stats.items():
#             row = {"model": model_name, "log_file": file_key, **file_stats}
#             summary_data.append(row)

#     # Write to JSON
#     if summary_data:  # Only write if there's data
#         try:
#             with open(output_file, "w", encoding="utf-8") as jsonfile:
#                 json.dump(summary_data, jsonfile, indent=2, default=str)

#             print(f"\nSaved summary statistics to: {output_file}")
#             return {"summary": summary_data}
#         except Exception as e:
#             print(f"Error saving summary: {e}")
#             return {}
#     else:
#         print("No summary data to save")
#         return {}
