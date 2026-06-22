#!/usr/bin/env python3
"""Run the localized RBF tracking experiment and save a renderable track."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from . import rbf_tracking as tracking
except ImportError:
    import rbf_tracking as tracking


def main(argv: list[str] | None = None) -> int:
    parser = tracking.build_parser()
    parser.description = "Run online localized RBF tracking and save a track."
    parser.set_defaults(output=tracking.DEFAULT_TRACK)
    args = parser.parse_args(argv)
    tracking.save_track(args, args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
