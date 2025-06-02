import os
import json
from collections import defaultdict


class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics = defaultdict(list)

    def log(self, key, value):
        self.metrics[key].append(value)

    def log_dict(self, data: dict):
        for k, v in data.items():
            self.log(k, v)

    def save(self, filename="log.json"):
        path = os.path.join(self.log_dir, filename)
        with open(path, "w") as f:
            json.dump(self.metrics, f)

    def get(self, key):
        return self.metrics[key]

    def all(self):
        return dict(self.metrics)
