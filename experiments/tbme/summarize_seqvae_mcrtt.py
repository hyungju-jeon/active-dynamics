#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from seqvae_mcrtt import DEFAULT_BASE_DIR, DEFAULT_CONFIG_PATH, load_config, summarize_session

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiment_common import resolve_session_root
else:
    from .seqvae_mcrtt import DEFAULT_BASE_DIR, DEFAULT_CONFIG_PATH, load_config, summarize_session
    from ..experiment_common import resolve_session_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the latest SeqVAE-on-MC_RTT TBME session.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    session_root = resolve_session_root(Path(args.base_dir), create=False, exp_ids=["seqvae_mcrtt"])
    return int(summarize_session(session_root=session_root, config=config))


if __name__ == "__main__":
    raise SystemExit(main())
