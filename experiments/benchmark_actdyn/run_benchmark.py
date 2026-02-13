#!/usr/bin/env python3
"""Convenience entrypoint for benchmark process + analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from experiments.benchmark_actdyn import analyze_benchmark, process_benchmark
else:  # pragma: no cover
    from . import analyze_benchmark, process_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark processing and analysis")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parent / "conf" / "config.yaml"),
        help="Path to benchmark config yaml",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override processing output dir")
    parser.add_argument("--run-name", type=str, default=None, help="Override processing run name")

    parser.add_argument("--analysis-only", action="store_true", help="Run analysis only")
    parser.add_argument("--process-only", action="store_true", help="Run processing only")
    parser.add_argument("--input-dir", type=str, default=None, help="Existing run dir for analysis-only mode")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.analysis_only and args.process_only:
        raise ValueError("--analysis-only and --process-only cannot be used together")

    run_dir: Path
    if args.analysis_only:
        if args.input_dir is None:
            raise ValueError("--input-dir is required with --analysis-only")
        run_dir = Path(args.input_dir).expanduser().resolve()
    else:
        run_dir = process_benchmark.run_benchmark(
            config_path=args.config,
            output_dir=args.output_dir,
            run_name=args.run_name,
        )

    if not args.process_only:
        analyze_benchmark.analyze_benchmark(input_dir=run_dir, output_dir=run_dir, make_plots=True)

    print(f"Benchmark run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
