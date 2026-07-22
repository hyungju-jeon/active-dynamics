"""Generic drawing primitives shared across experiments.

Layering rule: everything here takes arrays / callables plus an ``ax`` (or
builds a small standalone figure) and knows nothing about experiments —
no run directories, no suite/policy vocabulary, no CLI. Experiment-specific
loaders, themes, and figure assembly live under ``experiments/<name>/``.
"""

from actdyn.visualize.style import (
    apply_manuscript_figure_style,
    decorate_phase_space_axis,
    make_lognorm,
    manuscript_rc,
    manuscript_rc_params,
    set_matplotlib_style,
    style_manuscript_axis,
)
from actdyn.visualize.vectorfield import (
    RbfVectorFieldDynamics,
    compute_vector_field,
    create_grid,
    evaluate_vector_field_grid,
    plot_vector_field,
    vector_field_l2_error,
)
from actdyn.visualize.trajectory import (
    annotate_action_arrow,
    create_gradient_line,
    plot_rollout_latent_comparison,
    trace_index,
)
from actdyn.visualize.neural import (
    plot_observation_channels,
    plot_spike_train,
)

__all__ = [
    "RbfVectorFieldDynamics",
    "annotate_action_arrow",
    "apply_manuscript_figure_style",
    "compute_vector_field",
    "create_gradient_line",
    "create_grid",
    "decorate_phase_space_axis",
    "evaluate_vector_field_grid",
    "make_lognorm",
    "manuscript_rc",
    "manuscript_rc_params",
    "plot_observation_channels",
    "plot_rollout_latent_comparison",
    "plot_spike_train",
    "plot_vector_field",
    "set_matplotlib_style",
    "style_manuscript_axis",
    "trace_index",
    "vector_field_l2_error",
]
