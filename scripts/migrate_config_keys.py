#!/usr/bin/env python3
"""Migrate legacy actdyn config keys/values to canonical strict keys.

Default targets:
- experiments/active_embedding/conf/*.yaml
- experiments/ciss/conf/*.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


KEY_RENAMES = {
    "obs_hidden_dim": "obs_hidden_dims",
    "act_hidden_dim": "act_hidden_dims",
    "enc_hidden_dim": "enc_hidden_dims",
    "enc_rnn_hidden_dim": "enc_rnn_hidden_dims",
    "map_hidden_dim": "map_hidden_dims",
    "dyn_hidden_dim": "dyn_hidden_dims",
}

VALUE_RENAMES = {
    "loglinear": "log-linear",
    "nonlinear": "non-linear",
    "A-optimality": "a-optimality",
    "D-optimality": "d-optimality",
    "Ensemble_disagreement": "ensemble-disagreement",
}

DEFAULT_GLOBS = [
    "experiments/active_embedding/conf/*.yaml",
    "experiments/ciss/conf/*.yaml",
]


def _rename_value(value: Any) -> Any:
    if isinstance(value, str):
        return VALUE_RENAMES.get(value, value)
    if isinstance(value, list):
        return [_rename_value(v) for v in value]
    if isinstance(value, dict):
        return _rename_mapping(value)
    return value


def _rename_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in mapping.items():
        new_key = KEY_RENAMES.get(key, key)
        out[new_key] = _rename_value(value)
    return out


def migrate_file(path: Path) -> tuple[bool, str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return False, "empty"
    if not isinstance(data, dict):
        return False, "non-dict root"

    migrated = _rename_mapping(data)
    changed = migrated != data
    dumped = yaml.safe_dump(migrated, sort_keys=False, allow_unicode=False)
    return changed, dumped


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate actdyn YAML config keys/values.")
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Additional glob pattern(s) to include.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes in-place. Without this flag, runs as dry-run.",
    )
    args = parser.parse_args()

    patterns = DEFAULT_GLOBS + args.glob
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(Path(".").glob(pattern)))

    if not files:
        print("No matching config files found.")
        return 0

    changed_count = 0
    for path in files:
        changed, output = migrate_file(path)
        state = "CHANGED" if changed else "UNCHANGED"
        print(f"{state}: {path}")
        if changed:
            changed_count += 1
            if args.write:
                path.write_text(output, encoding="utf-8")

    print(f"Total files: {len(files)}, changed: {changed_count}, write={args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
