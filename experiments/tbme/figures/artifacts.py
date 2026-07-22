"""Artifact writing conventions for TBME figures.

Every figure writes into ``<suite>/experiment/{figures,tables}/``; CSV/text
sidecars are mirrored into each participating suite directory.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from actdyn.utils.experiment_runtime import write_trace_csv


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    write_trace_csv(path, list(rows), list(fields))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        out.append(path)
        seen.add(path)
    return out


def artifact_paths(
    suite_dirs: Sequence[Path],
    *,
    subdir: str,
    filename: str,
) -> list[Path]:
    return unique_paths(suite_dir / "experiment" / subdir / filename for suite_dir in suite_dirs)


def write_csv_artifacts(
    suite_dirs: Sequence[Path],
    *,
    filename: str,
    rows: Iterable[dict[str, Any]],
    fields: Sequence[str],
) -> list[Path]:
    paths = artifact_paths(suite_dirs, subdir="tables", filename=filename)
    row_list = list(rows)
    for path in paths:
        write_csv(path, row_list, fields)
    return paths


def write_text_artifacts(
    suite_dirs: Sequence[Path],
    *,
    filename: str,
    text: str,
) -> list[Path]:
    paths = artifact_paths(suite_dirs, subdir="tables", filename=filename)
    for path in paths:
        write_text(path, text)
    return paths


def copy_artifact(source_path: Path, paths: Sequence[Path]) -> list[Path]:
    for path in paths:
        if path == source_path:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, path)
    return list(paths)
