"""Thin wrapper that forwards to the unified actdyn CLI."""

from __future__ import annotations

import sys

from actdyn.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["sweep", *sys.argv[1:]]))
