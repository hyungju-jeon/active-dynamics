from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import *  # noqa: F401,F403
else:
    from ..experiment_common import *  # noqa: F401,F403
