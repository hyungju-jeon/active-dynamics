import os
import json
import csv
from typing import Any
import threading


class Logger:
    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()  # Thread safety for concurrent logging

    def log(self, key: str, value: Any):
        """Log a metric value with memory-efficient handling."""
        if key not in self.metrics:
            self.metrics[key] = []

        # Convert tensors immediately to prevent memory retention
        if hasattr(value, "numel") and hasattr(value, "item"):
            if value.numel() == 1:  # Single-element tensor
                processed_value = value.item()
            else:  # Multi-element tensor -> convert to list
                processed_value = (
                    value.detach().cpu().tolist() if hasattr(value, "detach") else value.tolist()
                )
        elif hasattr(value, "detach"):  # Other tensor types
            processed_value = value.detach().cpu().tolist()
        else:
            processed_value = value

        self.metrics[key].append(processed_value)

    def save(self, path: str, append=True) -> None:
        """Append current step's metrics to a JSON file for real-time monitoring.
        Improved with better error handling and memory efficiency.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Get the latest values for each metric
        current_step_data = {
            "step": len(self.metrics.get("reward", [])) if "reward" in self.metrics else 0
        }

        # Add latest metric values with proper type conversion
        for key, values in self.metrics.items():
            if values:  # If there are values for this metric
                value = values[-1]  # Get the latest value
                # Convert tensors to Python native types
                if hasattr(value, "item"):  # Single-element tensor
                    current_step_data[key] = value.item()
                elif hasattr(value, "tolist"):  # Multi-element tensor
                    current_step_data[key] = value.tolist()
                else:
                    current_step_data[key] = value

        # Read existing data or initialize empty list with proper error handling
        existing_data = []
        if os.path.exists(path) and os.path.getsize(path) > 0 and append:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                # Ensure existing_data is a list
                if not isinstance(existing_data, list):
                    existing_data = []
            except (json.JSONDecodeError, ValueError, IOError) as e:
                print(f"Warning: Failed to read {path}, starting fresh: {e}")
                existing_data = []

        # Append new data
        existing_data.append(current_step_data)

        # Write back to file with error handling
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)
        except IOError as e:
            print(f"Warning: Failed to write to {path}: {e}")

    def clear(self):
        """Clear all logged metrics."""
        with self._lock:
            self.metrics.clear()
