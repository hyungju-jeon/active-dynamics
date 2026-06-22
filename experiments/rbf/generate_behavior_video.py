#!/usr/bin/env python3
"""Render saved online localized RBF tracking results."""

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


def _numbered_video_output(output: Path) -> Path:
    if len(output.name) >= 4 and output.name[:2].isdigit() and output.name[2] == "_":
        return output
    used = [
        int(path.name[:2])
        for path in output.parent.glob("*.mp4")
        if len(path.name) >= 4 and path.name[:2].isdigit() and path.name[2] == "_"
    ]
    return output.with_name(f"{max(used, default=0) + 1:02d}_{output.name}")


def main(argv: list[str] | None = None) -> int:
    args = tracking.build_parser().parse_args(argv)
    if args.self_test:
        tracking.self_test()
        return 0
    output = args.output.resolve()
    if args.mode == "frame":
        print(tracking.render_frame(args, output))
        return 0
    tracking.render_video(args, _numbered_video_output(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
