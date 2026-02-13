# benchmark_actdyn

Scaffolded benchmark for latent-state / parameter inference baselines with lightweight active learning policies.

## What this contains
- `process_benchmark.py`: runs benchmark rollouts and writes unified step metrics.
- `analyze_benchmark.py`: reads benchmark logs, writes `summary_table.csv`, and saves plots under `figures/`.
- `run_benchmark.py`: convenience entrypoint for process + analysis.
- `benchmark_spec.md`: benchmark contract for v1.
- `conf/`: runnable configs (`config.yaml`, `smoke.yaml`).

## Quick start
```bash
python experiments/benchmark_actdyn/run_benchmark.py \
  --config experiments/benchmark_actdyn/conf/smoke.yaml
```

Artifacts are written to `results/benchmark_actdyn/smoke_v1`.

## Process-only
```bash
python experiments/benchmark_actdyn/process_benchmark.py \
  --config experiments/benchmark_actdyn/conf/smoke.yaml
```

## Analyze existing run
```bash
python experiments/benchmark_actdyn/analyze_benchmark.py \
  --input-dir results/benchmark_actdyn/smoke_v1
```

## Notes
- Environment tracks are intentionally lightweight placeholders for v1 smoke coverage.
- `TODO(FLEX-v2)` markers identify extension points for FLEX integration.
