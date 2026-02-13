from __future__ import annotations

from pathlib import Path

from actdyn.utils.runtime import configure_runtime, ensure_dir


def test_configure_runtime_returns_valid_device():
    device = configure_runtime(seed=123, device=None)
    assert device in {"cpu", "cuda", "mps"}


def test_ensure_dir_creates_directory(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert Path(result).exists()
    assert Path(result).is_dir()
