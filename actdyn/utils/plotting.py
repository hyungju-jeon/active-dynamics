"""Backward-compatibility shim — the implementations moved to ``actdyn.visualize``.

Import from ``actdyn.visualize`` (primitives) or ``actdyn.visualize.legacy``
(figure-level helpers) in new code.
"""

from actdyn.visualize import (
    RbfVectorFieldDynamics,
    annotate_action_arrow,
    apply_manuscript_figure_style,
    compute_vector_field,
    create_gradient_line,
    create_grid,
    decorate_phase_space_axis,
    evaluate_vector_field_grid,
    make_lognorm,
    manuscript_rc,
    manuscript_rc_params,
    plot_observation_channels,
    plot_rollout_latent_comparison,
    plot_spike_train,
    plot_vector_field,
    set_matplotlib_style,
    style_manuscript_axis,
    trace_index,
    vector_field_l2_error,
)
from actdyn.visualize.legacy import (
    compute_fisher_map,
    create_subplot,
    plot_current_state,
    plot_embedding_error_comparison,
    plot_per_dimension,
)

__all__ = [
    "RbfVectorFieldDynamics",
    "annotate_action_arrow",
    "apply_manuscript_figure_style",
    "compute_fisher_map",
    "compute_vector_field",
    "create_gradient_line",
    "create_grid",
    "create_subplot",
    "decorate_phase_space_axis",
    "evaluate_vector_field_grid",
    "make_lognorm",
    "manuscript_rc",
    "manuscript_rc_params",
    "plot_current_state",
    "plot_embedding_error_comparison",
    "plot_observation_channels",
    "plot_per_dimension",
    "plot_rollout_latent_comparison",
    "plot_spike_train",
    "plot_vector_field",
    "set_matplotlib_style",
    "style_manuscript_axis",
    "trace_index",
    "vector_field_l2_error",
]
