"""TBME figure theme: visual style plus policy series presentation.

Two distinct concerns live here, deliberately separated:

- Visual style (colors of strokes/grids, rcParams application) — knows nothing
  about policies or suites.
- Series presentation — how TBME policy ids map to labels, colors, and sort
  order. Individual figures may override these locally; these maps are the
  shared defaults, not a contract every figure must obey.
"""

from __future__ import annotations

from typing import Any

from actdyn.visualize import apply_manuscript_figure_style, style_manuscript_axis

# Visual style
STROKE_COLOR = "#3A3A3A"
GRID_COLOR = "#DDD7CE"
NEUTRAL_LIGHT = "#C8C1B8"
NEUTRAL_FILL = "#F4F1EC"


def apply_style(plt_module: Any | None = None) -> None:
    if plt_module is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt_module
    apply_manuscript_figure_style(plt_module, stroke_color=STROKE_COLOR)


def style_axis(
    ax: Any,
    *,
    grid_axis: str | None = None,
    grid_color: str = GRID_COLOR,
    grid_alpha: float = 0.42,
) -> None:
    style_manuscript_axis(
        ax,
        grid_axis=grid_axis,
        grid_color=grid_color,
        grid_alpha=float(grid_alpha),
        stroke_color=STROKE_COLOR,
    )


def style_experiment_axis(ax: Any) -> None:
    style_axis(ax, grid_alpha=0.55)


# Series presentation
POLICY_LABELS = {
    "adaptive": "PALDI",
    "adaptive_async_anytime": "Async PALDI",
    "adaptive_async_realtime": "Async PALDI (zero-fill)",
    "active_planning": "Fixed PALDI",
    "active_myopic": "Myopic",
    "active_fully_observable": "Full obs.",
    "active_state_information": "State info",
    "active_dynamics": "Dyn. sens.",
    "active_e_optimality": "E-opt.",
    "active_observation_variance": "Obs. var.",
    "active_state_variance": "State var.",
    "prbs": "PRBS",
    "random": "Random",
    "flex": "FLEX",
    "flex_true_state": "FLEX true state",
    "flex_filter": "FLEX upstream / filtered",
    "flex_true": "FLEX upstream / true",
    "flex_rollback": "FLEX rollback / filtered",
    "rhc": "RHC-US",
    "off_policy": "Off-policy",
    "active_planning_u1_r1_h40": "Planning u1/r1",
    "active_planning_u5_r5_h40": "Planning u5/r5",
    "active_planning_u1_r5_h40": "Planning u1/r5",
    "active_planning_u10_r10_h40": "Planning u10/r10",
    "active_planning_u5_r10_h40": "Planning u5/r10",
    "active_planning_u5_r20_h40": "Planning u5/r20",
    "active_planning_u10_r20_h40": "Planning u10/r20",
}

POLICY_ORDER = [
    "active_planning",
    "active_planning_u1_r1_h40",
    "active_planning_u1_r5_h40",
    "active_planning_u5_r5_h40",
    "active_planning_u5_r10_h40",
    "active_planning_u10_r10_h40",
    "active_planning_u5_r20_h40",
    "active_planning_u10_r20_h40",
    "adaptive",
    "adaptive_async_realtime",
    "adaptive_async_anytime",
    "adaptive_state_fixed_update",
    "adaptive_state",
    "active_fully_observable",
    "active_e_optimality",
    "active_state_information",
    "active_dynamics",
    "active_observation_variance",
    "active_state_variance",
    "active_myopic",
    "prbs",
    "random",
    "flex",
    "flex_true_state",
    "flex_filter",
    "flex_true",
    "flex_rollback",
    "rhc",
]

POLICY_COLORS = {
    "active_planning": "#5DADE2",
    "active_planning_u1_r1_h40": "#85C1E9",
    "active_planning_u5_r5_h40": "#73C6B6",
    "active_planning_u1_r5_h40": "#82E0AA",
    "active_planning_u10_r10_h40": "#BB8FCE",
    "active_planning_u5_r10_h40": "#76D7C4",
    "active_planning_u5_r20_h40": "#AED6F1",
    "active_planning_u10_r20_h40": "#7FB3D5",
    "adaptive": "#F1948A",
    "adaptive_async_realtime": "#C85C5C",
    "adaptive_async_anytime": "#8E4B7D",
    "adaptive_state_fixed_update": "#D7BDE2",
    "adaptive_state": "#F8C471",
    "active_fully_observable": "#82E0AA",
    "active_e_optimality": "#BB8FCE",
    "active_state_information": "#F7DC6F",
    "active_dynamics": "#76D7C4",
    "active_observation_variance": "#D2B48C",
    "active_myopic": "#F5B041",
    "prbs": "#45B8AC",
    "random": "#9EA7AD",
    "flex": "#AF7AC5",
    "flex_true_state": "#58D68D",
    "flex_filter": "#2E86C1",
    "flex_true": "#239B56",
    "flex_rollback": "#CB4335",
    "active_state_variance": "#58D68D",
    "rhc": "#F06292",
}
FALLBACK_COLORS = (
    "#5DADE2",
    "#F1948A",
    "#58D68D",
    "#AF7AC5",
    "#F5B041",
    "#9EA7AD",
    "#45B8AC",
    "#F7DC6F",
)

_SHORT_POLICY_LABELS = {
    "active_planning": "Planning",
    "active_fully_observable": "Full obs.",
    "active_e_optimality": "E-opt.",
    "active_state_information": "State info",
    "active_dynamics": "Dynamics",
    "active_observation_variance": "Obs. var.",
    "active_myopic": "Myopic",
    "active_state_variance": "State var.",
    "prbs": "PRBS",
    "random": "Random",
}


def policy_sort_key(policy_id: str) -> tuple[int, str]:
    try:
        return POLICY_ORDER.index(policy_id), policy_id
    except ValueError:
        return len(POLICY_ORDER), policy_id


def policy_label(policy_id: str) -> str:
    return POLICY_LABELS.get(policy_id, policy_id.replace("_", " "))


def policy_color(policy_id: str, fallback_idx: int = 0) -> str:
    return POLICY_COLORS.get(policy_id, FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)])


def short_policy_label(policy_id: str) -> str:
    return _SHORT_POLICY_LABELS.get(
        policy_id, POLICY_LABELS.get(policy_id, policy_id.replace("_", " "))
    )
