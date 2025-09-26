import os
import json
import pickle
from typing import Dict, List, Any, Optional
from pathlib import Path

from actdyn.config import ExperimentConfig
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


def save_config(config: ExperimentConfig, path=None):
    """Save experiment config to disk."""
    if path is None:
        path = config.results_dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config.to_yaml(os.path.join(path, "config.yaml"))


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
