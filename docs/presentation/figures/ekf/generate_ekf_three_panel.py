from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any
from typing import Iterable
from typing import Sequence
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Patch


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_SEED = 17
DEFAULT_BASENAME = "ekf_three_panel"
DEFAULT_DPI = 300
DEFAULT_FORMATS = "svg,png"
SUPPORTED_FORMATS = {"png", "svg", "pdf"}


STYLE = {
    "state": "#387ef5",
    "state_dark": "#1f4ea4",
    "predicted_var": "#d24646",
    "measurement": "#7d7d7d",
    "innovation": "#d24646",
    "actual_update": "#2e8b57",
    "vector": "#c9c9c9",
    "text": "#2f2f2f",
}

TEXT_SIZE = 11.5
TITLE_SIZE = 17.0
PANEL_XLIM = (-2.75, -0.55)
PANEL_YLIM = (-1.75, 0.25)


def nonlinear_transition(state: np.ndarray) -> np.ndarray:
    """Nonlinear dynamics used to build the prediction panel geometry."""
    x, y = state
    return np.array(
        [
            0.92 * x + 0.55 * y + 0.15 * np.sin(0.9 * x),
            0.22 * x + 0.95 * y + 0.12 * np.cos(0.8 * y),
        ],
        dtype=float,
    )


def transition_jacobian(state: np.ndarray) -> np.ndarray:
    """Jacobian of the nonlinear transition evaluated at one state."""
    x, y = state
    return np.array(
        [
            [0.92 + 0.135 * np.cos(0.9 * x), 0.55],
            [0.22, 0.95 - 0.096 * np.sin(0.8 * y)],
        ],
        dtype=float,
    )


def covariance_to_ellipse_params(
    covariance: np.ndarray,
    n_std: float = 2.0,
) -> tuple[float, float, float]:
    """Convert a 2x2 covariance matrix into width/height/rotation for an ellipse."""
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (2, 2):
        raise ValueError("Covariance must be shape (2, 2).")

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.clip(eigvals[order], 1e-12, None)
    eigvecs = eigvecs[:, order]

    width, height = 2.0 * n_std * np.sqrt(eigvals)
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    return float(width), float(height), angle


def compute_ekf_geometry(seed: int = DEFAULT_SEED, params: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
    """Build one-step EKF geometry for an illustrative 2D latent state."""
    config = {
        "mu0": np.array([-1.25, -0.75], dtype=float),
        "P0": np.array([[0.10, 0.025], [0.025, 0.07]], dtype=float),
        "Q": np.array([[0.02, 0.005], [0.005, 0.015]], dtype=float),
        "R": np.array([[0.09, 0.018], [0.018, 0.065]], dtype=float),
    }
    if params:
        for key, value in params.items():
            config[key] = np.asarray(value, dtype=float)

    rng = np.random.default_rng(seed)
    mu0 = config["mu0"]
    P0 = config["P0"]
    Q = config["Q"]
    R = config["R"]

    F = transition_jacobian(mu0)
    mu_pred = nonlinear_transition(mu0)
    P_pred = F @ P0 @ F.T + Q

    H = np.eye(2, dtype=float)
    measurement_bias = np.array([0.32, -0.22], dtype=float)
    measurement_noise = rng.multivariate_normal(np.zeros(2, dtype=float), 0.18 * R)
    measurement = mu_pred + measurement_bias + measurement_noise

    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    innovation = measurement - H @ mu_pred

    mu_post = mu_pred + K @ innovation
    P_post = (np.eye(2) - K @ H) @ P_pred
    P_post = 0.5 * (P_post + P_post.T)

    return {
        "mu0": mu0,
        "P0": P0,
        "Q": Q,
        "F": F,
        "mu_pred": mu_pred,
        "P_pred": P_pred,
        "R": R,
        "measurement": measurement,
        "innovation": innovation,
        "K": K,
        "mu_post": mu_post,
        "P_post": P_post,
    }


def _add_covariance_ellipse(
    ax: Axes,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    edgecolor: str,
    facecolor: str = "none",
    alpha: float = 0.2,
    linewidth: float = 2.0,
    linestyle: str = "-",
    n_std: float = 2.0,
    gid: str | None = None,
) -> Ellipse:
    width, height, angle = covariance_to_ellipse_params(covariance, n_std=n_std)
    ellipse = Ellipse(
        xy=(float(mean[0]), float(mean[1])),
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=3,
    )
    if gid:
        ellipse.set_gid(gid)
    ax.add_patch(ellipse)
    return ellipse


def _setup_panel(ax: Axes, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=8.0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#fdfdfd")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#b7b7b7")


def _build_vector_field_grid(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[np.ndarray, ...]:
    xs = np.linspace(xlim[0], xlim[1], 24)
    ys = np.linspace(ylim[0], ylim[1], 24)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    flat = np.column_stack([X.ravel(), Y.ravel()])
    next_states = np.array([nonlinear_transition(point) for point in flat])
    delta = next_states - flat
    U = delta[:, 0].reshape(X.shape)
    V = delta[:, 1].reshape(Y.shape)
    return X, Y, U, V


def _draw_vector_field_background(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    style: dict[str, str],
    density: float = 0.95,
    alpha: float = 0.75,
) -> None:
    X, Y, U, V = _build_vector_field_grid(xlim, ylim)
    stream = ax.streamplot(
        X,
        Y,
        U,
        V,
        density=density,
        color=style["vector"],
        linewidth=0.65,
        arrowsize=0.60,
        zorder=1,
    )
    stream.lines.set_alpha(alpha)
    stream.arrows.set_alpha(alpha)
    stream.lines.set_gid("vector-field")
    stream.arrows.set_gid("vector-field")


def _add_annotation_block(
    ax: Axes,
    lines: list[str],
    style: dict[str, str],
    x: float = 0.03,
    y_start: float = 0.95,
    y_step: float = 0.09,
) -> None:
    for idx, text in enumerate(lines):
        ax.text(
            x,
            y_start - idx * y_step,
            text,
            transform=ax.transAxes,
            fontsize=TEXT_SIZE,
            fontweight="bold",
            color=style["text"],
            va="top",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.86,
            },
            zorder=10,
        )


def draw_panel_prediction(
    ax: Axes,
    geom: dict[str, np.ndarray],
    style: dict[str, str] = STYLE,
    xlim: tuple[float, float] = PANEL_XLIM,
    ylim: tuple[float, float] = PANEL_YLIM,
) -> None:
    _setup_panel(ax, "Prediction", xlim=xlim, ylim=ylim)
    _draw_vector_field_background(ax, xlim, ylim, style, density=0.92, alpha=0.82)

    _add_covariance_ellipse(
        ax,
        geom["mu0"],
        geom["P0"],
        edgecolor="none",
        facecolor=style["state"],
        alpha=0.24,
        linewidth=0.0,
        linestyle="-",
        gid="initial-belief-ellipse",
    )
    _add_covariance_ellipse(
        ax,
        geom["mu_pred"],
        geom["P_pred"],
        edgecolor="none",
        facecolor=style["predicted_var"],
        alpha=0.15,
        linewidth=0.0,
        linestyle=":",
        gid="predictive-ellipse",
    )

    ax.scatter(*geom["mu0"], s=42, color=style["state_dark"], zorder=5)
    ax.scatter(*geom["mu_pred"], s=48, color=style["state"], zorder=5)
    ax.text(
        geom["mu0"][0] + 0.10,
        geom["mu0"][1] + 0.30,
        r"Initial point $\hat{x}_{t-1}$",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["state_dark"],
        zorder=10,
    )
    ax.text(
        geom["mu_pred"][0] - 0.44,
        geom["mu_pred"][1] - 0.26,
        r"Predictive mean $\hat{x}_{t}^{-}$",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["state_dark"],
        zorder=10,
    )

    arrow = FancyArrowPatch(
        posA=tuple(geom["mu0"]),
        posB=tuple(geom["mu_pred"]),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        linestyle="--",
        color=style["state_dark"],
        zorder=5,
    )
    arrow.set_gid("prior-to-prediction-arrow")
    ax.add_patch(arrow)

    _add_annotation_block(
        ax,
        lines=[
            r"$\hat{x}_{t}^{-}=f(\hat{x}_{t-1})$",
            r"$P_t^{-}=F_tP_{t-1}F_t^\top+Q_t$",
        ],
        style=style,
    )


def draw_panel_measurement(
    ax: Axes,
    geom: dict[str, np.ndarray],
    style: dict[str, str] = STYLE,
    xlim: tuple[float, float] = PANEL_XLIM,
    ylim: tuple[float, float] = PANEL_YLIM,
) -> None:
    _setup_panel(ax, "Measurement", xlim=xlim, ylim=ylim)
    _draw_vector_field_background(ax, xlim, ylim, style, density=0.92, alpha=0.65)

    _add_covariance_ellipse(
        ax,
        geom["mu_pred"],
        geom["P_pred"],
        edgecolor="none",
        facecolor=style["predicted_var"],
        alpha=0.09,
        linewidth=0.0,
        linestyle=":",
        gid="predictive-reference-ellipse",
    )
    _add_covariance_ellipse(
        ax,
        geom["measurement"],
        geom["R"],
        edgecolor="none",
        facecolor=style["measurement"],
        alpha=0.24,
        linewidth=0.0,
        gid="measurement-uncertainty-ellipse",
    )
    ax.scatter(*geom["mu_pred"], s=40, color=style["state"], zorder=5)
    ax.scatter(*geom["measurement"], s=52, color=style["measurement"], zorder=6)
    ax.text(
        geom["measurement"][0] + 0.10,
        geom["measurement"][1] - 0.16,
        r"Projected target",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["measurement"],
        zorder=10,
    )

    innovation_arrow = FancyArrowPatch(
        posA=tuple(geom["mu_pred"]),
        posB=tuple(geom["measurement"]),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.1,
        color=style["innovation"],
        zorder=7,
    )
    innovation_arrow.set_gid("innovation-arrow")
    ax.add_patch(innovation_arrow)

    midpoint = 0.5 * (geom["mu_pred"] + geom["measurement"])
    ax.text(
        midpoint[0] + 0.12,
        midpoint[1] + 0.28,
        r"$y_t-h(\hat{x}_{t}^{-})$",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["innovation"],
        zorder=10,
    )
    _add_annotation_block(
        ax,
        lines=[
            r"innovation from $\hat{x}_{t}^{-}$",
            r"measurement pulls belief",
        ],
        style=style,
    )


def draw_panel_update(
    ax: Axes,
    geom: dict[str, np.ndarray],
    style: dict[str, str] = STYLE,
    xlim: tuple[float, float] = PANEL_XLIM,
    ylim: tuple[float, float] = PANEL_YLIM,
) -> None:
    _setup_panel(ax, "Update", xlim=xlim, ylim=ylim)
    _draw_vector_field_background(ax, xlim, ylim, style, density=0.92, alpha=0.65)

    _add_covariance_ellipse(
        ax,
        geom["mu_post"],
        geom["P_post"],
        edgecolor="none",
        facecolor=style["state"],
        alpha=0.28,
        linewidth=0.0,
        gid="posterior-ellipse",
    )

    ax.scatter(*geom["mu_pred"], s=36, color=style["state"], alpha=0.6, zorder=5)
    ax.scatter(*geom["mu_post"], s=54, color=style["state_dark"], zorder=6)
    projected_target = ax.scatter(
        *geom["measurement"],
        s=44,
        color=style["measurement"],
        zorder=6,
    )
    projected_target.set_gid("projected-target-update")
    ax.text(
        geom["mu_post"][0] + 0.25,
        geom["mu_post"][1] - 0.28,
        r"Posterior mean $\hat{x}_{t}$",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["state_dark"],
        zorder=10,
    )
    ax.text(
        geom["measurement"][0] + 0.08,
        geom["measurement"][1] + 0.14,
        r"Projected target",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["measurement"],
        zorder=10,
    )

    update_arrow = FancyArrowPatch(
        posA=tuple(geom["mu_pred"]),
        posB=tuple(geom["mu_post"]),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=2.1,
        color=style["actual_update"],
        zorder=7,
    )
    update_arrow.set_gid("update-arrow")
    ax.add_patch(update_arrow)

    midpoint = 0.5 * (geom["mu_pred"] + geom["mu_post"])
    ax.text(
        midpoint[0] + 0.1,
        midpoint[1] + 0.3,
        r"$K_t\!\left(y_t-h(\hat{x}_{t}^{-})\right)$",
        fontsize=TEXT_SIZE,
        fontweight="bold",
        color=style["actual_update"],
        zorder=10,
    )
    _add_annotation_block(
        ax,
        lines=[
            r"$\hat{x}_t=\hat{x}_t^{-}+K_t(y_t-h(\hat{x}_t^{-}))$",
            r"$P_t=(I-K_t)P_t^{-}$",
        ],
        style=style,
    )


def _add_figure_legend(fig: Figure) -> None:
    handles = [
        Line2D([0], [0], color=STYLE["vector"], lw=1.3, label="Vector field"),
        Patch(
            facecolor=STYLE["state"],
            edgecolor="none",
            alpha=0.24,
            label="Initial covariance",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["predicted_var"],
            lw=2.0,
            linestyle=":",
            label="Predicted covariance (dotted)",
        ),
        Patch(
            facecolor=STYLE["state"],
            edgecolor="none",
            alpha=0.28,
            label="Posterior covariance",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["measurement"],
            lw=0.0,
            marker="o",
            markersize=6.5,
            label="Projected target",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["innovation"],
            lw=2.2,
            marker=">",
            markevery=[1],
            label="Innovation direction",
        ),
        Line2D(
            [0],
            [0],
            color=STYLE["actual_update"],
            lw=2.2,
            marker=">",
            markevery=[1],
            label="Actual update direction",
        ),
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        framealpha=0.92,
        edgecolor="#d0d0d0",
        fontsize=TEXT_SIZE,
        bbox_to_anchor=(0.5, 0.035),
    )
    legend_text_colors = {
        "Vector field": STYLE["vector"],
        "Initial covariance": STYLE["state_dark"],
        "Predicted covariance (dotted)": STYLE["predicted_var"],
        "Posterior covariance": STYLE["state_dark"],
        "Projected target": STYLE["measurement"],
        "Innovation direction": STYLE["innovation"],
        "Actual update direction": STYLE["actual_update"],
    }
    for text in legend.get_texts():
        text.set_fontweight("bold")
        label = text.get_text()
        if label in legend_text_colors:
            text.set_color(legend_text_colors[label])


def build_figure(
    seed: int = DEFAULT_SEED,
    params: dict[str, Any] | None = None,
    figsize: tuple[float, float] = (12.8, 7.2),
) -> Figure:
    """Build the 3-panel EKF illustration figure."""
    geom = compute_ekf_geometry(seed=seed, params=params)
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)

    draw_panel_prediction(axes[0], geom, STYLE)
    draw_panel_measurement(axes[1], geom, STYLE)
    draw_panel_update(axes[2], geom, STYLE)
    fig.suptitle(
        "Extended Kalman Filter: Predict, Measure, Update",
        fontsize=TITLE_SIZE + 2.0,
        y=0.97,
    )
    _add_figure_legend(fig)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.88, bottom=0.17, wspace=0.12)
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


def _svg_text_content(text_element: ET.Element) -> str:
    parts: list[str] = []
    if text_element.text:
        parts.append(text_element.text)
    for child in text_element:
        if child.text:
            parts.append(child.text)
    merged = " ".join("".join(parts).split())
    return merged.strip()


def _slugify_text(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug if slug else "text"


def _group_svg_text_by_content(svg_path: Path) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    if root.tag.startswith("{"):
        svg_namespace = root.tag[1:].split("}", 1)[0]
        ET.register_namespace("", svg_namespace)
    else:
        svg_namespace = "http://www.w3.org/2000/svg"

    xlink_namespace = "http://www.w3.org/1999/xlink"
    ET.register_namespace("xlink", xlink_namespace)

    text_nodes: list[ET.Element] = [
        node for node in root.iter() if node.tag == f"{{{svg_namespace}}}text"
    ]
    by_content: dict[str, list[ET.Element]] = {}
    for node in text_nodes:
        text_value = _svg_text_content(node)
        if text_value:
            by_content.setdefault(text_value, []).append(node)

    if not any(len(nodes) >= 2 for nodes in by_content.values()):
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        return

    group_container = ET.Element(f"{{{svg_namespace}}}g")
    group_container.set("id", "text-groups-index")
    group_container.set("style", "display:none")

    used_ids: set[str] = {node.attrib["id"] for node in root.iter() if "id" in node.attrib}
    used_group_ids: set[str] = set()
    node_id_counter = 1

    for text_value, nodes in by_content.items():
        if len(nodes) < 2:
            continue

        base_group_id = f"text-group-{_slugify_text(text_value)}"
        group_id = base_group_id
        suffix = 2
        while group_id in used_group_ids:
            group_id = f"{base_group_id}-{suffix}"
            suffix += 1
        used_group_ids.add(group_id)

        group = ET.Element(f"{{{svg_namespace}}}g")
        group.set("id", group_id)
        group.set("data-text", text_value)

        for node in nodes:
            if "id" in node.attrib:
                node_id = node.attrib["id"]
            else:
                node_id = f"text-node-{node_id_counter}"
                while node_id in used_ids:
                    node_id_counter += 1
                    node_id = f"text-node-{node_id_counter}"
                node.set("id", node_id)
                used_ids.add(node_id)
                node_id_counter += 1

            existing_class = node.attrib.get("class", "")
            class_tokens = [token for token in existing_class.split() if token]
            if group_id not in class_tokens:
                class_tokens.append(group_id)
                node.set("class", " ".join(class_tokens))

            use_node = ET.Element(f"{{{svg_namespace}}}use")
            use_node.set("href", f"#{node_id}")
            use_node.set(f"{{{xlink_namespace}}}href", f"#{node_id}")
            group.append(use_node)

        group_container.append(group)

    root.append(group_container)

    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def _group_svg_vector_fields(svg_path: Path) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    if root.tag.startswith("{"):
        svg_namespace = root.tag[1:].split("}", 1)[0]
        ET.register_namespace("", svg_namespace)
    else:
        svg_namespace = "http://www.w3.org/2000/svg"

    xlink_namespace = "http://www.w3.org/1999/xlink"
    ET.register_namespace("xlink", xlink_namespace)

    vector_nodes: list[ET.Element] = []
    for node in root.iter():
        node_id = node.attrib.get("id", "")
        node_class = node.attrib.get("class", "")
        class_tokens = node_class.split()
        if node_id == "vector-field" or "vector-field" in class_tokens:
            vector_nodes.append(node)

    if not vector_nodes:
        tree.write(svg_path, encoding="utf-8", xml_declaration=True)
        return

    used_ids: set[str] = {node.attrib["id"] for node in root.iter() if "id" in node.attrib}
    seen_ids: set[str] = set()
    for idx, node in enumerate(vector_nodes, start=1):
        node_id = node.attrib.get("id")
        if (not node_id) or (node_id in seen_ids):
            candidate = f"vector-field-{idx}"
            suffix = 2
            while candidate in used_ids:
                candidate = f"vector-field-{idx}-{suffix}"
                suffix += 1
            node_id = candidate
            node.set("id", node_id)
            used_ids.add(node_id)
        seen_ids.add(node_id)

        class_tokens = node.attrib.get("class", "").split()
        if "vector-field" not in class_tokens:
            class_tokens.append("vector-field")
            node.set("class", " ".join(class_tokens))

    vector_group = ET.Element(f"{{{svg_namespace}}}g")
    vector_group.set("id", "vector-fields-group")
    vector_group.set("style", "display:none")
    vector_group.set("data-role", "vector-fields-index")

    for node in vector_nodes:
        node_id = node.attrib["id"]
        use_node = ET.Element(f"{{{svg_namespace}}}use")
        use_node.set("href", f"#{node_id}")
        use_node.set(f"{{{xlink_namespace}}}href", f"#{node_id}")
        vector_group.append(use_node)

    root.append(vector_group)
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def save_figure(
    fig: Figure,
    outdir: str | Path,
    basename: str = DEFAULT_BASENAME,
    formats: str | Iterable[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
) -> list[Path]:
    """Save the EKF figure to one or more output formats."""
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    selected_formats = _parse_formats(formats)

    saved_paths: list[Path] = []
    for file_format in selected_formats:
        output_path = outdir_path / f"{basename}.{file_format}"
        if file_format == "svg":
            with plt.rc_context({"svg.fonttype": "none"}):
                fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
            _group_svg_text_by_content(output_path)
            _group_svg_vector_fields(output_path)
        else:
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(output_path)
    return saved_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a 3-panel EKF illustration (prediction, measurement, update)."
    )
    parser.add_argument(
        "--outdir",
        default="docs/presentation/figures/ekf",
        help="Directory where figure files are written.",
    )
    parser.add_argument(
        "--basename",
        default=DEFAULT_BASENAME,
        help="Base filename for saved outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help="Raster DPI used for PNG export.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for deterministic measurement sampling.",
    )
    parser.add_argument(
        "--formats",
        default=DEFAULT_FORMATS,
        help="Comma-separated output formats. Supported: png, svg, pdf.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    fig = build_figure(seed=args.seed)
    try:
        output_paths = save_figure(
            fig=fig,
            outdir=args.outdir,
            basename=args.basename,
            formats=args.formats,
            dpi=args.dpi,
        )
    finally:
        plt.close(fig)

    for path in output_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
