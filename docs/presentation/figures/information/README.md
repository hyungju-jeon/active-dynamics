# Information Components Figure

This directory contains a standalone figure generator for a `2x2` matrix illustrating two components of information in Bayesian recursive learning:

- `Sensitivity`: how strongly the signal changes
- `Certainty`: how reliable the signal is

The matrix classifies four cases:

- `Uninformative`
- `Informative`
- `Inconclusive evidence`
- `Unreliable information`

## Generate the Figure

From the repository root:

```bash
python docs/presentation/figures/information/generate_information_sensitivity_certainty.py
```

Default outputs:

- `docs/presentation/figures/information/information_sensitivity_certainty.svg`
- `docs/presentation/figures/information/information_sensitivity_certainty.png`

## Optional Flags

```bash
python docs/presentation/figures/information/generate_information_sensitivity_certainty.py \
  --outdir docs/presentation/figures/information \
  --basename information_sensitivity_certainty \
  --dpi 300 \
  --formats svg,png
```
