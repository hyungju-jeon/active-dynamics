from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse
from matplotlib.transforms import Bbox
import numpy as np
import pytest


matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "docs" / "presentation" / "figures" / "generate_ekf_three_panel.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("ekf_three_panel", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_ekf_geometry_is_deterministic():
    module = _load_module()
    first = module.compute_ekf_geometry(seed=11)
    second = module.compute_ekf_geometry(seed=11)

    for key in ("mu0", "mu_pred", "mu_post", "measurement", "P0", "P_pred", "P_post"):
        assert np.allclose(first[key], second[key], atol=1e-10)


def test_posterior_covariance_is_psd_and_contracts():
    module = _load_module()
    geom = module.compute_ekf_geometry(seed=5)

    eigvals = np.linalg.eigvalsh(geom["P_post"])
    assert np.all(eigvals >= -1e-10)

    pred_trace = float(np.trace(geom["P_pred"]))
    post_trace = float(np.trace(geom["P_post"]))
    assert post_trace < pred_trace


def test_default_uncertainty_is_compact():
    module = _load_module()
    geom = module.compute_ekf_geometry(seed=5)

    assert float(np.trace(geom["P0"])) <= 0.20
    assert float(np.trace(geom["R"])) <= 0.20
    assert float(np.trace(geom["P_pred"])) <= 0.45


def test_covariance_to_ellipse_params_returns_positive_axes():
    module = _load_module()
    cov = np.array([[0.35, 0.08], [0.08, 0.20]])
    width, height, angle = module.covariance_to_ellipse_params(cov, n_std=2.0)

    assert width > 0.0
    assert height > 0.0
    assert isinstance(angle, float)


def test_build_figure_has_three_panels_and_expected_titles():
    module = _load_module()
    fig = module.build_figure(seed=3)
    try:
        titles = [ax.get_title() for ax in fig.axes]
        assert titles == ["Prediction", "Measurement", "Update"]
    finally:
        module.plt.close(fig)


def test_panels_are_zoomed_to_informative_region():
    module = _load_module()
    fig = module.build_figure(seed=3)
    try:
        for ax in fig.axes:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            xspan = float(xlim[1] - xlim[0])
            yspan = float(ylim[1] - ylim[0])
            assert xspan <= 2.3
            assert yspan <= 2.1
    finally:
        module.plt.close(fig)


def test_all_panels_include_vector_field():
    module = _load_module()
    fig = module.build_figure(seed=3)
    try:
        for ax in fig.axes:
            has_vector_field = any(
                getattr(artist, "get_gid", lambda: None)() == "vector-field"
                for artist in ax.get_children()
            )
            assert has_vector_field
    finally:
        module.plt.close(fig)


def _bbox_overlaps(a: Bbox, b: Bbox) -> bool:
    return (
        a.x0 < b.x1
        and a.x1 > b.x0
        and a.y0 < b.y1
        and a.y1 > b.y0
    )


def test_panel_text_boxes_do_not_overlap():
    module = _load_module()
    fig = module.build_figure(seed=6)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for ax in fig.axes:
            text_boxes: list[Bbox] = []
            for text in ax.texts:
                bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
                text_boxes.append(bbox)
            for i, first in enumerate(text_boxes):
                for second in text_boxes[i + 1 :]:
                    assert not _bbox_overlaps(first, second)
    finally:
        module.plt.close(fig)


def test_text_font_sizes_are_larger_than_default():
    module = _load_module()
    fig = module.build_figure(seed=7)
    try:
        min_text_size = min(
            text.get_fontsize()
            for ax in fig.axes
            for text in ax.texts
        )
        min_title_size = min(ax.title.get_fontsize() for ax in fig.axes)
        assert min_text_size >= 11.0
        assert min_title_size >= 15.0
    finally:
        module.plt.close(fig)


def test_panel_text_is_bold():
    module = _load_module()
    fig = module.build_figure(seed=7)
    try:
        for ax in fig.axes:
            assert str(ax.title.get_fontweight()) in {"bold", "700"}
            for text in ax.texts:
                assert str(text.get_fontweight()) in {"bold", "700"}
    finally:
        module.plt.close(fig)


def test_required_semantic_labels_present():
    module = _load_module()
    geom = module.compute_ekf_geometry(seed=10)
    fig = module.build_figure(seed=10)
    try:
        def _assert_label_near_point(ax, fragment: str, point: np.ndarray, max_dist: float = 0.55):
            matching = [text for text in ax.texts if fragment in text.get_text()]
            assert matching, f"Missing label containing '{fragment}'"
            distances = []
            for text in matching:
                tx, ty = text.get_position()
                distances.append(float(np.linalg.norm(np.array([tx, ty]) - point)))
            assert min(distances) <= max_dist, f"Label '{fragment}' not placed near point"

        prediction_ax = fig.axes[0]
        measurement_ax = fig.axes[1]
        update_ax = fig.axes[2]
        _assert_label_near_point(prediction_ax, "Initial point", geom["mu0"])
        _assert_label_near_point(prediction_ax, "Predictive mean", geom["mu_pred"])
        _assert_label_near_point(measurement_ax, "Projected target", geom["measurement"])
        _assert_label_near_point(update_ax, "Posterior mean", geom["mu_post"])
    finally:
        module.plt.close(fig)


def test_covariance_styles_and_update_visibility():
    module = _load_module()
    fig = module.build_figure(seed=10)
    try:
        prediction_ax = fig.axes[0]
        measurement_ax = fig.axes[1]
        update_ax = fig.axes[2]

        initial_cov = next(
            (
                patch
                for patch in prediction_ax.patches
                if isinstance(patch, Ellipse) and patch.get_gid() == "initial-belief-ellipse"
            ),
            None,
        )
        predictive_cov = next(
            (
                patch
                for patch in prediction_ax.patches
                if isinstance(patch, Ellipse) and patch.get_gid() == "predictive-ellipse"
            ),
            None,
        )
        predictive_cov_measurement = next(
            (
                patch
                for patch in measurement_ax.patches
                if isinstance(patch, Ellipse) and patch.get_gid() == "predictive-reference-ellipse"
            ),
            None,
        )
        posterior_cov = next(
            (
                patch
                for patch in update_ax.patches
                if isinstance(patch, Ellipse) and patch.get_gid() == "posterior-ellipse"
            ),
            None,
        )
        measurement_cov = next(
            (
                patch
                for patch in measurement_ax.patches
                if isinstance(patch, Ellipse) and patch.get_gid() == "measurement-uncertainty-ellipse"
            ),
            None,
        )

        assert initial_cov is not None
        assert predictive_cov is not None
        assert predictive_cov_measurement is not None
        assert measurement_cov is not None
        assert posterior_cov is not None
        for cov in [
            initial_cov,
            predictive_cov,
            predictive_cov_measurement,
            measurement_cov,
            posterior_cov,
        ]:
            edge = np.array(to_rgba(cov.get_edgecolor()))
            assert cov.get_linewidth() <= 0.01
            assert edge[-1] <= 1e-6

        update_covariances = [patch for patch in update_ax.patches if isinstance(patch, Ellipse)]
        assert len(update_covariances) == 1
    finally:
        module.plt.close(fig)


def test_measurement_panel_contains_innovation_arrow():
    module = _load_module()
    fig = module.build_figure(seed=4)
    try:
        measurement_ax = fig.axes[1]
        found = False
        for artist in measurement_ax.get_children():
            if getattr(artist, "get_gid", lambda: None)() == "innovation-arrow":
                found = True
                break
        assert found
    finally:
        module.plt.close(fig)


def test_figure_contains_legend():
    module = _load_module()
    fig = module.build_figure(seed=12)
    try:
        assert len(fig.legends) >= 1
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        assert "Vector field" in labels
    finally:
        module.plt.close(fig)


def test_legend_text_is_bold_and_color_matched():
    module = _load_module()
    fig = module.build_figure(seed=12)
    try:
        legend = fig.legends[0]
        expected_colors = {
            "Initial covariance": module.STYLE["state_dark"],
            "Predicted covariance (dotted)": module.STYLE["predicted_var"],
            "Projected target": module.STYLE["measurement"],
            "Actual update direction": module.STYLE["actual_update"],
        }

        for text in legend.get_texts():
            label = text.get_text()
            if label in expected_colors:
                assert text.get_fontweight() in {"bold", "semibold", 700}
                assert np.allclose(
                    np.array(to_rgba(text.get_color())),
                    np.array(to_rgba(expected_colors[label])),
                    atol=1e-3,
                )
    finally:
        module.plt.close(fig)


def test_update_panel_uses_distinct_update_color_and_shows_projected_target():
    module = _load_module()
    geom = module.compute_ekf_geometry(seed=11)
    fig = module.build_figure(seed=11)
    try:
        measurement_ax = fig.axes[1]
        update_ax = fig.axes[2]

        innovation_arrow = next(
            (
                artist for artist in measurement_ax.patches
                if getattr(artist, "get_gid", lambda: None)() == "innovation-arrow"
            ),
            None,
        )
        update_arrow = next(
            (
                artist for artist in update_ax.patches
                if getattr(artist, "get_gid", lambda: None)() == "update-arrow"
            ),
            None,
        )
        assert innovation_arrow is not None
        assert update_arrow is not None
        assert not np.allclose(
            np.array(to_rgba(update_arrow.get_edgecolor())),
            np.array(to_rgba(innovation_arrow.get_edgecolor())),
            atol=1e-3,
        )

        projected_target_scatter = next(
            (
                artist for artist in update_ax.collections
                if getattr(artist, "get_gid", lambda: None)() == "projected-target-update"
            ),
            None,
        )
        assert projected_target_scatter is not None

        target_texts = [t for t in update_ax.texts if "Projected target" in t.get_text()]
        assert target_texts
        distances = [
            float(np.linalg.norm(np.array(t.get_position()) - geom["measurement"]))
            for t in target_texts
        ]
        assert min(distances) <= 0.55
    finally:
        module.plt.close(fig)


def test_updated_mean_differs_from_predictive_mean():
    module = _load_module()
    geom = module.compute_ekf_geometry(seed=8)
    assert not np.allclose(geom["mu_post"], geom["mu_pred"])


def test_posterior_ellipse_contracts_vs_predictive():
    module = _load_module()
    geom = module.compute_ekf_geometry(seed=9)
    pred_w, pred_h, _ = module.covariance_to_ellipse_params(geom["P_pred"], n_std=2.0)
    post_w, post_h, _ = module.covariance_to_ellipse_params(geom["P_post"], n_std=2.0)
    assert post_w < pred_w
    assert post_h < pred_h


def test_save_figure_writes_png_and_svg(tmp_path: Path):
    module = _load_module()
    fig = module.build_figure(seed=2)
    try:
        paths = module.save_figure(
            fig=fig,
            outdir=tmp_path,
            basename="ekf_three_panel_test",
            formats="svg,png",
            dpi=150,
        )
    finally:
        module.plt.close(fig)

    assert len(paths) == 2
    assert (tmp_path / "ekf_three_panel_test.svg").exists()
    assert (tmp_path / "ekf_three_panel_test.png").exists()


def test_svg_output_groups_identical_text_labels(tmp_path: Path):
    module = _load_module()
    fig = module.build_figure(seed=2)
    try:
        paths = module.save_figure(
            fig=fig,
            outdir=tmp_path,
            basename="ekf_three_panel_test",
            formats="svg",
            dpi=150,
        )
    finally:
        module.plt.close(fig)

    assert len(paths) == 1
    svg_path = tmp_path / "ekf_three_panel_test.svg"
    tree = ET.parse(svg_path)
    root = tree.getroot()

    grouped_nodes = [
        node
        for node in root.iter()
        if node.tag.endswith("g") and node.attrib.get("id", "").startswith("text-group-")
    ]
    assert grouped_nodes

    projected_target_groups = [
        node for node in grouped_nodes if node.attrib.get("data-text") == "Projected target"
    ]
    assert projected_target_groups
    grouped_use_children = [
        child for child in projected_target_groups[0] if child.tag.endswith("use")
    ]
    assert len(grouped_use_children) >= 2


def test_svg_output_groups_vector_fields_together(tmp_path: Path):
    module = _load_module()
    fig = module.build_figure(seed=2)
    try:
        module.save_figure(
            fig=fig,
            outdir=tmp_path,
            basename="ekf_three_panel_test",
            formats="svg",
            dpi=150,
        )
    finally:
        module.plt.close(fig)

    svg_path = tmp_path / "ekf_three_panel_test.svg"
    tree = ET.parse(svg_path)
    root = tree.getroot()

    vector_group = next(
        (
            node
            for node in root.iter()
            if node.tag.endswith("g") and node.attrib.get("id") == "vector-fields-group"
        ),
        None,
    )
    assert vector_group is not None
    grouped_use_children = [child for child in vector_group if child.tag.endswith("use")]
    assert len(grouped_use_children) >= 3


def test_save_figure_rejects_unsupported_format(tmp_path: Path):
    module = _load_module()
    fig = module.build_figure(seed=2)
    try:
        with pytest.raises(ValueError, match="Unsupported format"):
            module.save_figure(
                fig=fig,
                outdir=tmp_path,
                basename="ekf_three_panel_test",
                formats="gif",
                dpi=100,
            )
    finally:
        module.plt.close(fig)
