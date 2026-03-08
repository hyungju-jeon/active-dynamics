# EKF Three-Panel Figure

This directory contains a standalone script that generates an Extended Kalman Filter illustration with three panels:

- Prediction
- Measurement
- Update

## Generate the Figure

From the repository root:

```bash
python docs/presentation/figures/ekf/generate_ekf_three_panel.py
```

Default outputs:

- `docs/presentation/figures/ekf/ekf_three_panel.svg`
- `docs/presentation/figures/ekf/ekf_three_panel.png`

## Optional Flags

```bash
python docs/presentation/figures/ekf/generate_ekf_three_panel.py \
  --outdir docs/presentation/figures/ekf \
  --basename ekf_three_panel \
  --dpi 300 \
  --seed 17 \
  --formats svg,png
```
