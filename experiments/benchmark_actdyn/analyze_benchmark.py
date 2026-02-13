#!/usr/bin/env python3
"""Analyze benchmark metrics and produce summary tables + plots."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

FLOAT_FIELDS = {
    "theta_true",
    "theta_hat",
    "param_abs_error",
    "latent_abs_error",
    "posterior_var",
    "info_gain",
    "reward",
    "action",
    "action_norm",
    "policy_cost",
    "runtime_ms",
}
INT_FIELDS = {"seed", "episode", "step"}


def _load_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _parse_row(row: dict[str, str]) -> dict[str, Any]:
    parsed: dict[str, Any] = dict(row)
    for key in INT_FIELDS:
        if key in parsed and parsed[key] != "":
            parsed[key] = int(parsed[key])
    for key in FLOAT_FIELDS:
        if key in parsed and parsed[key] != "":
            parsed[key] = float(parsed[key])
    return parsed


def load_metrics(input_dir: Path) -> list[dict[str, Any]]:
    metrics_csv = input_dir / "metrics.csv"
    metrics_jsonl = input_dir / "metrics.jsonl"

    if metrics_csv.exists():
        with metrics_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [_parse_row(dict(row)) for row in reader]

    if metrics_jsonl.exists():
        rows: list[dict[str, Any]] = []
        with metrics_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    rows.append(_parse_row({k: str(v) for k, v in row.items()}))
        return rows

    raise FileNotFoundError(f"No metrics file found in: {input_dir}")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _group_episode_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["env_name"]), str(row["method"]), int(row["seed"]), int(row["episode"]))
        grouped[key].append(row)

    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["step"]))
    return grouped


def summarize_episode_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_episode_rows(rows)
    episode_rows: list[dict[str, Any]] = []

    for (env_name, method, seed, episode), records in sorted(grouped.items()):
        if not records:
            continue

        final = records[-1]
        episode_rows.append(
            {
                "env_name": env_name,
                "method": method,
                "seed": seed,
                "episode": episode,
                "steps": len(records),
                "final_param_abs_error": float(final["param_abs_error"]),
                "final_latent_abs_error": float(final["latent_abs_error"]),
                "final_posterior_var": float(final["posterior_var"]),
                "mean_info_gain": float(np.mean([r["info_gain"] for r in records])),
                "mean_reward": float(np.mean([r["reward"] for r in records])),
                "mean_action_norm": float(np.mean([r["action_norm"] for r in records])),
                "mean_runtime_ms": float(np.mean([r["runtime_ms"] for r in records])),
            }
        )

    return episode_rows


def summarize_table(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        key = (str(row["env_name"]), str(row["method"]))
        grouped[key].append(row)

    summary: list[dict[str, Any]] = []
    for (env_name, method), records in sorted(grouped.items()):
        final_param = np.asarray([r["final_param_abs_error"] for r in records], dtype=np.float64)
        final_latent = np.asarray([r["final_latent_abs_error"] for r in records], dtype=np.float64)
        info_gain = np.asarray([r["mean_info_gain"] for r in records], dtype=np.float64)
        reward = np.asarray([r["mean_reward"] for r in records], dtype=np.float64)
        runtime_ms = np.asarray([r["mean_runtime_ms"] for r in records], dtype=np.float64)

        summary.append(
            {
                "env_name": env_name,
                "method": method,
                "n_runs": len(records),
                "final_param_abs_error_mean": float(final_param.mean()),
                "final_param_abs_error_std": float(final_param.std()),
                "final_latent_abs_error_mean": float(final_latent.mean()),
                "final_latent_abs_error_std": float(final_latent.std()),
                "mean_info_gain_mean": float(info_gain.mean()),
                "mean_info_gain_std": float(info_gain.std()),
                "mean_reward_mean": float(reward.mean()),
                "mean_reward_std": float(reward.std()),
                "mean_runtime_ms_mean": float(runtime_ms.mean()),
                "mean_runtime_ms_std": float(runtime_ms.std()),
            }
        )

    return summary


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _write_line_svg(
    series: dict[str, tuple[list[float], list[float]]],
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    width, height = 900, 480
    left, right, top, bottom = 70, 30, 40, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_x = [x for xs, _ in series.values() for x in xs]
    all_y = [y for _, ys in series.values() for y in ys]
    if not all_x or not all_y:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    if abs(x_max - x_min) < 1e-12:
        x_max = x_min + 1.0
    if abs(y_max - y_min) < 1e-12:
        y_max = y_min + 1.0

    def map_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def map_y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#17becf"]
    content: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />",
        f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-size='18'>{_svg_escape(title)}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333' />",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333' />",
    ]

    for idx, (name, (xs, ys)) in enumerate(sorted(series.items())):
        points = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(xs, ys))
        color = colors[idx % len(colors)]
        content.append(
            f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.2' />"
        )
        legend_y = top + 18 + idx * 18
        content.append(
            f"<rect x='{left + plot_w - 180}' y='{legend_y - 9}' width='10' height='10' fill='{color}' />"
        )
        content.append(
            f"<text x='{left + plot_w - 165}' y='{legend_y}' font-size='12'>{_svg_escape(name)}</text>"
        )

    content.extend(
        [
            f"<text x='{width/2:.1f}' y='{height - 20}' text-anchor='middle' font-size='13'>{_svg_escape(xlabel)}</text>",
            (
                f"<text x='22' y='{height/2:.1f}' transform='rotate(-90 22,{height/2:.1f})' "
                f"text-anchor='middle' font-size='13'>{_svg_escape(ylabel)}</text>"
            ),
            "</svg>",
        ]
    )

    out_path.write_text("\n".join(content), encoding="utf-8")


def _write_bar_svg(labels: list[str], values: list[float], out_path: Path, title: str, ylabel: str) -> None:
    width, height = 900, 480
    left, right, top, bottom = 70, 30, 40, 95
    plot_w = width - left - right
    plot_h = height - top - bottom

    if not labels:
        out_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    y_max = max(values) if values else 1.0
    y_max = 1.0 if y_max <= 0 else y_max * 1.1

    bar_width = plot_w / max(len(labels), 1)
    content: list[str] = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' />",
        f"<text x='{width/2:.1f}' y='24' text-anchor='middle' font-size='18'>{_svg_escape(title)}</text>",
        f"<line x1='{left}' y1='{top + plot_h}' x2='{left + plot_w}' y2='{top + plot_h}' stroke='#333' />",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_h}' stroke='#333' />",
    ]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = left + idx * bar_width + bar_width * 0.15
        w = bar_width * 0.7
        h = (value / y_max) * plot_h
        y = top + plot_h - h
        content.append(f"<rect x='{x:.2f}' y='{y:.2f}' width='{w:.2f}' height='{h:.2f}' fill='#1f77b4' />")
        content.append(
            f"<text x='{x + w/2:.2f}' y='{top + plot_h + 18}' text-anchor='middle' font-size='11' transform='rotate(20 {x + w/2:.2f},{top + plot_h + 18})'>{_svg_escape(label)}</text>"
        )

    content.extend(
        [
            (
                f"<text x='22' y='{height/2:.1f}' transform='rotate(-90 22,{height/2:.1f})' "
                f"text-anchor='middle' font-size='13'>{_svg_escape(ylabel)}</text>"
            ),
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(content), encoding="utf-8")


def _plot_param_error_curves(rows: list[dict[str, Any]], figures_dir: Path) -> Path:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        step = int(row["step"])
        grouped[(method, step)].append(float(row["param_abs_error"]))

    methods = sorted({method for method, _ in grouped})
    plot_series: dict[str, tuple[list[float], list[float]]] = {}
    for method in methods:
        steps = sorted({step for m, step in grouped if m == method})
        means = [float(np.mean(grouped[(method, step)])) for step in steps]
        plot_series[method] = (steps, means)

    plt = _load_matplotlib()
    if plt is not None:
        plt.figure(figsize=(8, 4.5))
        for method, (steps, means) in plot_series.items():
            stds = np.asarray([np.std(grouped[(method, step)]) for step in steps], dtype=np.float64)
            means_np = np.asarray(means, dtype=np.float64)
            steps_np = np.asarray(steps, dtype=np.float64)
            plt.plot(steps_np, means_np, label=method)
            plt.fill_between(steps_np, means_np - stds, means_np + stds, alpha=0.15)

        plt.title("Parameter Error Over Steps")
        plt.xlabel("Step")
        plt.ylabel("Absolute Parameter Error")
        plt.grid(alpha=0.2)
        plt.legend(fontsize=8)
        plt.tight_layout()

        out_path = figures_dir / "param_error_over_steps.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        return out_path

    out_path = figures_dir / "param_error_over_steps.svg"
    _write_line_svg(
        plot_series,
        out_path=out_path,
        title="Parameter Error Over Steps",
        xlabel="Step",
        ylabel="Absolute Parameter Error",
    )
    return out_path


def _plot_final_param_bars(summary_rows: list[dict[str, Any]], figures_dir: Path) -> Path:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in summary_rows:
        grouped[str(row["method"])].append(float(row["final_param_abs_error_mean"]))

    methods = sorted(grouped)
    means = [float(np.mean(grouped[method])) for method in methods]

    plt = _load_matplotlib()
    if plt is not None:
        plt.figure(figsize=(8, 4.5))
        plt.bar(methods, means)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("Final Parameter Error (lower is better)")
        plt.title("Final Parameter Error by Method")
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()

        out_path = figures_dir / "final_param_error_by_method.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        return out_path

    out_path = figures_dir / "final_param_error_by_method.svg"
    _write_bar_svg(
        labels=methods,
        values=means,
        out_path=out_path,
        title="Final Parameter Error by Method",
        ylabel="Final Parameter Error (lower is better)",
    )
    return out_path


def analyze_benchmark(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    make_plots: bool = True,
) -> dict[str, Path]:
    source_dir = Path(input_dir).expanduser().resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Benchmark run directory not found: {source_dir}")

    target_dir = source_dir if output_dir is None else Path(output_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    rows = load_metrics(source_dir)
    episode_rows = summarize_episode_rows(rows)
    summary_rows = summarize_table(episode_rows)

    summary_path = target_dir / "summary_table.csv"
    _write_csv(
        summary_path,
        summary_rows,
        fields=[
            "env_name",
            "method",
            "n_runs",
            "final_param_abs_error_mean",
            "final_param_abs_error_std",
            "final_latent_abs_error_mean",
            "final_latent_abs_error_std",
            "mean_info_gain_mean",
            "mean_info_gain_std",
            "mean_reward_mean",
            "mean_reward_std",
            "mean_runtime_ms_mean",
            "mean_runtime_ms_std",
        ],
    )

    episode_path = target_dir / "episode_table.csv"
    _write_csv(
        episode_path,
        episode_rows,
        fields=[
            "env_name",
            "method",
            "seed",
            "episode",
            "steps",
            "final_param_abs_error",
            "final_latent_abs_error",
            "final_posterior_var",
            "mean_info_gain",
            "mean_reward",
            "mean_action_norm",
            "mean_runtime_ms",
        ],
    )

    outputs: dict[str, Path] = {
        "summary_table": summary_path,
        "episode_table": episode_path,
    }

    if make_plots:
        figures_dir = target_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        outputs["plot_param_error"] = _plot_param_error_curves(rows, figures_dir)
        outputs["plot_final_param"] = _plot_final_param_bars(summary_rows, figures_dir)

    # TODO(FLEX-v2): Add FLEX-specific calibration / regret figures once FLEX baseline is integrated.

    print(f"Benchmark analysis complete: {source_dir}")
    print(f"- summary table: {summary_path}")
    if make_plots:
        print(f"- figures dir: {target_dir / 'figures'}")

    return outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze benchmark outputs")
    parser.add_argument("--input-dir", type=str, required=True, help="Benchmark run directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional analysis output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    analyze_benchmark(input_dir=args.input_dir, output_dir=args.output_dir, make_plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
