# FitzHugh-Nagumo and Hopf Design

**Goal:** Add FitzHugh-Nagumo and Hopf vector fields to the core environment and expose two-parameter versions in the COSYNE information-map GUI.

## Scope

- Add `fitzhugh_nagumo` and `hopf` to the core `VectorFieldEnv` registry.
- Implement full parameterizations in `actdyn/utils/vectorfield_definition.py`.
- Extend `experiments/cosyne/info_map_gui.py` to support both dynamics through the existing two-slider interface.
- Add tests for core dynamics evaluation and GUI parsing/evaluation.

## Core Environment Design

### FitzHugh-Nagumo

Use the two-dimensional system

- `x_dot = x - x^3 / 3 - y + i_ext`
- `y_dot = (x + a - b * y) / tau`

Expose the full parameter set `(a, b, tau, i_ext)` through the core environment class. Default values should be stable canonical values that produce interpretable phase portraits without extra tuning.

### Hopf

Use the generalized Hopf normal form

- `r2 = x^2 + y^2`
- `x_dot = (mu - r2) * x - (omega + beta * r2) * y`
- `y_dot = (omega + beta * r2) * x + (mu - r2) * y`

Expose the full parameter set `(mu, omega, beta)` through the core environment class. Keep `beta=0.0` as the default so the default behavior matches the standard Hopf oscillator.

## GUI Design

Keep the existing two-parameter GUI surface and map the sliders to fixed reduced parameterizations:

- FitzHugh-Nagumo:
  - `param_a -> i_ext`
  - `param_b -> a`
  - fix `b=0.8`, `tau=12.5`
- Hopf:
  - `param_a -> omega`
  - `param_b -> mu`
  - fix `beta=0.0`

Extend the selector aliases and labels so both names work in CLI and radio-button selection. Add per-dynamics slider presets so the default view is informative after switching models, while preserving the current two-slider layout.

## Testing Design

- Add core tests that instantiate both environments, set parameters, and compare `compute_dynamics()` to the closed-form equations at known states.
- Extend GUI tests to verify parser aliases and `compute_dynamics_velocity()` support both new dynamics.
- Follow TDD: write failing tests first, confirm failure, implement the minimal code, then rerun targeted tests.
