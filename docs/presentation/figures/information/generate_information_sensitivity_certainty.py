from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable
from typing import Sequence

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_BASENAME = "information_sensitivity_certainty"
DEFAULT_DPI = 300
DEFAULT_FORMATS = "svg,png"
SUPPORTED_FORMATS = {"png", "svg", "pdf"}

SIGNAL_X = np.linspace(0.05, 0.95, 300)
OBSERVATION_X = np.array([0.16, 0.28, 0.42, 0.56, 0.70, 0.82], dtype=float)
OBSERVATION_PATTERN = np.array([-0.42, 0.16, -0.08, 0.22, -0.10, 0.18], dtype=float)
ERROR_PATTERN = np.array([0.90, 0.80, 0.76, 0.84, 0.78, 0.86], dtype=float)
ERROR_BAR_SCALE = 1.4
LOW_CERTAINTY_ERROR_BAR_SCALE = 1.8
LOW_CERTAINTY_OBSERVATION_OFFSET_SCALE = 1.7
PLOT_LEFT = 0.14
PLOT_BOTTOM = 0.20
PLOT_WIDTH = 0.76
PLOT_HEIGHT = 0.62

COLORS = {
    "text": "#27323b",
    "axis": "#727b84",
    "signal": "#8fa8bd",
    "signal_shadow": "#c9ced1",
    "observation_fill": "#5a4d62",
    "observation_edge": "#31404a",
    "card_accent": "#6d7783",
    "card_background": "#eef1f4",
    "shadow": "#d7dcd9",
}


def get_case_definitions() -> list[dict[str, object]]:
    """Return row-major information taxonomy definitions for the 2x2 matrix."""
    return [
        {
            "case_id": "low_sensitivity_high_certainty",
            "row": 0,
            "col": 0,
            "label": "Uninformative",
            "descriptor": "Flat but precise",
            "sensitivity_level": "low",
            "certainty_level": "high",
            "slope": 1.6,
            "curve_low": 0.495,
            "curve_high": 0.525,
            "noise_scale": 0.030,
            "accent": COLORS["card_accent"],
            "background": COLORS["card_background"],
            "highlight": False,
        },
        {
            "case_id": "high_sensitivity_high_certainty",
            "row": 0,
            "col": 1,
            "label": "Informative",
            "descriptor": "Steep and precise",
            "sensitivity_level": "high",
            "certainty_level": "high",
            "slope": 10.0,
            "curve_low": 0.18,
            "curve_high": 0.88,
            "noise_scale": 0.040,
            "accent": COLORS["card_accent"],
            "background": COLORS["card_background"],
            "highlight": True,
        },
        {
            "case_id": "low_sensitivity_low_certainty",
            "row": 1,
            "col": 0,
            "label": "Inconclusive evidence",
            "descriptor": "Flat and noisy",
            "sensitivity_level": "low",
            "certainty_level": "low",
            "slope": 1.6,
            "curve_low": 0.495,
            "curve_high": 0.525,
            "noise_scale": 0.11,
            "accent": COLORS["card_accent"],
            "background": COLORS["card_background"],
            "highlight": False,
        },
        {
            "case_id": "high_sensitivity_low_certainty",
            "row": 1,
            "col": 1,
            "label": "Unreliable information",
            "descriptor": "Steep but noisy",
            "sensitivity_level": "high",
            "certainty_level": "low",
            "slope": 10.0,
            "curve_low": 0.18,
            "curve_high": 0.88,
            "noise_scale": 0.11,
            "accent": COLORS["card_accent"],
            "background": COLORS["card_background"],
            "highlight": False,
        },
    ]


def compute_case_plot_data(case: dict[str, object]) -> dict[str, np.ndarray]:
    """Return deterministic signal and observation arrays for one case.

    The returned dictionary contains:
    - `signal_x`: shape `(300,)`
    - `signal_y`: shape `(300,)`
    - `observation_x`: shape `(6,)`
    - `observation_y`: shape `(6,)`
    - `observation_error`: shape `(6,)`
    """
    slope = float(case["slope"])
    curve_low = float(case["curve_low"])
    curve_high = float(case["curve_high"])
    noise_scale = float(case["noise_scale"])
    certainty_level = str(case["certainty_level"])

    logistic = 1.0 / (1.0 + np.exp(-slope * (SIGNAL_X - 0.55)))
    signal_y = curve_low + (curve_high - curve_low) * logistic

    observation_logistic = 1.0 / (1.0 + np.exp(-slope * (OBSERVATION_X - 0.55)))
    observation_signal = curve_low + (curve_high - curve_low) * observation_logistic
    offset_scale = 1.0 if certainty_level == "high" else LOW_CERTAINTY_OBSERVATION_OFFSET_SCALE
    error_scale = ERROR_BAR_SCALE if certainty_level == "high" else LOW_CERTAINTY_ERROR_BAR_SCALE
    observation_y = np.clip(
        observation_signal + offset_scale * noise_scale * OBSERVATION_PATTERN,
        0.10,
        0.90,
    )
    observation_error = error_scale * noise_scale * ERROR_PATTERN

    return {
        "signal_x": SIGNAL_X,
        "signal_y": np.clip(signal_y, 0.07, 0.93),
        "observation_x": OBSERVATION_X,
        "observation_y": observation_y,
        "observation_error": observation_error,
    }


def _to_response_coordinates(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_panel = PLOT_LEFT + PLOT_WIDTH * x_values
    y_panel = PLOT_BOTTOM + PLOT_HEIGHT * y_values
    return x_panel, y_panel


def _style_case_axis(ax, case: dict[str, object]) -> None:
    ax.set_gid(str(case["case_id"]))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("none")
    ax.set_aspect("equal", adjustable="box")
    ax.patch.set_alpha(0.0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    shadow = FancyBboxPatch(
        (0.01, -0.01),
        1.0,
        1.0,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        transform=ax.transAxes,
        linewidth=0.0,
        facecolor=COLORS["shadow"],
        alpha=0.12,
        zorder=-30,
        clip_on=False,
    )
    shadow.set_gid(f"shadow-{case['case_id']}")
    ax.add_patch(shadow)

    card = FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.015,rounding_size=0.03",
        transform=ax.transAxes,
        linewidth=1.55 if bool(case["highlight"]) else 1.25,
        edgecolor=str(case["accent"]),
        facecolor=str(case["background"]),
        zorder=-20,
        clip_on=False,
    )
    card.set_gid(f"card-{case['case_id']}")
    ax.add_patch(card)

    x_axis = FancyArrowPatch(
        (PLOT_LEFT, PLOT_BOTTOM),
        (PLOT_LEFT + PLOT_WIDTH, PLOT_BOTTOM),
        arrowstyle="simple,head_length=8,head_width=6,tail_width=0.8",
        color=COLORS["axis"],
        linewidth=0.0,
        mutation_scale=1,
        transform=ax.transAxes,
        zorder=2,
    )
    x_axis.set_gid(f"mini-xaxis-{case['case_id']}")
    ax.add_patch(x_axis)

    y_axis = FancyArrowPatch(
        (PLOT_LEFT, PLOT_BOTTOM),
        (PLOT_LEFT, PLOT_BOTTOM + PLOT_HEIGHT),
        arrowstyle="simple,head_length=8,head_width=6,tail_width=0.8",
        color=COLORS["axis"],
        linewidth=0.0,
        mutation_scale=1,
        transform=ax.transAxes,
        zorder=2,
    )
    y_axis.set_gid(f"mini-yaxis-{case['case_id']}")
    ax.add_patch(y_axis)

    ax.text(
        0.04,
        0.92,
        str(case["descriptor"]),
        transform=ax.transAxes,
        fontsize=8.7,
        fontweight="semibold",
        color=str(case["accent"]),
        va="top",
    )
    ax.text(
        0.50,
        0.07,
        str(case["label"]),
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="semibold",
        color=str(case["accent"]),
        ha="center",
        va="center",
    )
    ax.text(
        PLOT_LEFT + 0.5 * PLOT_WIDTH,
        PLOT_BOTTOM - 0.055,
        "Input perturbation",
        transform=ax.transAxes,
        fontsize=7.3,
        color=COLORS["axis"],
        ha="center",
        va="top",
    )
    ax.text(
        PLOT_LEFT - 0.08,
        PLOT_BOTTOM + 0.5 * PLOT_HEIGHT,
        "Expected signal",
        transform=ax.transAxes,
        fontsize=7.3,
        color=COLORS["axis"],
        ha="center",
        va="center",
        rotation=90,
    )
    ax.text(
        0.70,
        0.80,
        "Observation uncertainty",
        transform=ax.transAxes,
        fontsize=7.3,
        color=COLORS["axis"],
        ha="center",
        va="center",
    )


def _draw_case_content(ax, case: dict[str, object]) -> None:
    data = compute_case_plot_data(case)
    signal_x, signal_y = _to_response_coordinates(data["signal_x"], data["signal_y"])
    observation_x, observation_y = _to_response_coordinates(
        data["observation_x"], data["observation_y"]
    )
    observation_error = PLOT_HEIGHT * data["observation_error"]

    ax.plot(
        signal_x,
        signal_y,
        color=COLORS["signal_shadow"],
        linewidth=4.8,
        alpha=0.17,
        solid_capstyle="round",
        zorder=2.2,
    )

    signal_line = ax.plot(
        signal_x,
        signal_y,
        color=COLORS["signal"],
        linewidth=2.35,
        solid_capstyle="round",
        zorder=2.5,
    )[0]
    signal_line.set_gid(f"signal-{case['case_id']}")

    uncertainty = ax.vlines(
        observation_x,
        observation_y - observation_error,
        observation_y + observation_error,
        color=COLORS["observation_fill"],
        linewidth=1.3,
        alpha=0.82,
        zorder=2.7,
    )
    uncertainty.set_gid(f"uncertainty-{case['case_id']}")

    observations = ax.scatter(
        observation_x,
        observation_y,
        s=32,
        color=COLORS["observation_fill"],
        edgecolors=COLORS["observation_edge"],
        linewidths=0.9,
        zorder=3.0,
    )
    observations.set_gid(f"observations-{case['case_id']}")


def _add_dimension_annotations(fig: Figure) -> None:
    sensitivity_arrow = FancyArrowPatch(
        (0.18, 0.08),
        (0.88, 0.08),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=COLORS["axis"],
        transform=fig.transFigure,
    )
    sensitivity_arrow.set_gid("sensitivity-axis-arrow")
    fig.add_artist(sensitivity_arrow)

    certainty_arrow = FancyArrowPatch(
        (0.08, 0.17),
        (0.08, 0.84),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.6,
        color=COLORS["axis"],
        transform=fig.transFigure,
    )
    certainty_arrow.set_gid("certainty-axis-arrow")
    fig.add_artist(certainty_arrow)

    fig.text(
        0.50,
        0.95,
        "Two Components of Information in Bayesian Recursive Learning",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(
        0.50,
        0.91,
        "Information requires both sensitivity and certainty.",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["text"],
    )
    fig.text(
        0.50,
        0.03,
        "Sensitivity: How strongly does the signal change?",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
    )
    fig.text(0.16, 0.03, "Low", ha="center", va="center", fontsize=9, color=COLORS["axis"])
    fig.text(0.88, 0.03, "High", ha="center", va="center", fontsize=9, color=COLORS["axis"])
    fig.text(
        0.02,
        0.50,
        "Certainty: How reliable is that signal?",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=COLORS["text"],
        rotation=90,
    )
    fig.text(0.02, 0.16, "Low", ha="center", va="center", fontsize=9, color=COLORS["axis"], rotation=90)
    fig.text(0.02, 0.84, "High", ha="center", va="center", fontsize=9, color=COLORS["axis"], rotation=90)


def build_figure(figsize: tuple[float, float] = (8.0, 8.0)) -> Figure:
    """Build the 2x2 sensitivity-certainty information figure."""
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    fig.patch.set_facecolor("white")

    axis_positions = {
        (0, 0): [0.15, 0.52, 0.32, 0.32],
        (0, 1): [0.53, 0.52, 0.32, 0.32],
        (1, 0): [0.15, 0.15, 0.32, 0.32],
        (1, 1): [0.53, 0.15, 0.32, 0.32],
    }

    for case in get_case_definitions():
        ax = fig.add_axes(axis_positions[(int(case["row"]), int(case["col"]))])
        _style_case_axis(ax, case)
        _draw_case_content(ax, case)

    _add_dimension_annotations(fig)
    return fig


def _parse_formats(formats: str | Iterable[str]) -> list[str]:
    if isinstance(formats, str):
        requested = [part.strip().lower() for part in formats.split(",") if part.strip()]
    else:
        requested = [str(part).strip().lower() for part in formats if str(part).strip()]

    if not requested:
        raise ValueError("No output formats requested.")

    unsupported = sorted(set(requested) - SUPPORTED_FORMATS)
    if unsupported:
        raise ValueError(
            f"Unsupported format(s): {', '.join(unsupported)}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )
    return requested


def save_figure(
    fig: Figure,
    outdir: str | Path,
    basename: str = DEFAULT_BASENAME,
    formats: str | Iterable[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Save the figure to the requested output directory and formats."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    selected_formats = _parse_formats(formats)

    saved_paths: list[Path] = []
    for file_format in selected_formats:
        output_path = outdir_path / f"{basename}.{file_format}"
        if file_format == "svg":
            with plt.rc_context({"svg.fonttype": "none"}):
                fig.savefig(output_path, dpi=dpi)
        else:
            fig.savefig(output_path, dpi=dpi)
        saved_paths.append(output_path)
    return saved_paths


def main(argv: Sequence[str] | None = None) -> list[Path]:
    """Parse CLI arguments, generate the figure, and save it to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("docs/presentation/figures/information"),
        help="Directory where output figures are written.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=DEFAULT_BASENAME,
        help="Base filename for saved outputs.",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Rasterization DPI.")
    parser.add_argument(
        "--formats",
        type=str,
        default=DEFAULT_FORMATS,
        help="Comma-separated output formats.",
    )
    args = parser.parse_args(argv)

    fig = build_figure()
    try:
        return save_figure(
            fig=fig,
            outdir=args.outdir,
            basename=args.basename,
            formats=args.formats,
            dpi=args.dpi,
        )
    finally:
        plt.close(fig)


if __name__ == "__main__":
    for path in main():
        print(path)
