#!/usr/bin/env python3
"""Backward-compatibility shim — implementation moved to experiments/tbme/figures/summary.py."""
from __future__ import annotations

from experiments.tbme.figures.summary import (  # noqa: F401
    make_summary_metric_figures,
    make_summary_trajectory_figures,
    summary_main,
)
