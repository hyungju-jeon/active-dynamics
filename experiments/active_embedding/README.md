# Active Embedding Scripts

The active-embedding workflow is split into processing and analysis entrypoints.

## Processing (data generation / training)

```bash
python3 experiments/active_embedding/process_active_embedding.py
```

This runs `exp_active.py` with `run_analysis=False` and writes rollouts/results under `results/active_embedding`.

## Analysis (post-processing)

```bash
python3 experiments/active_embedding/analyze_active_embedding.py \
  --base-dir results/active_embedding
```

By default this expects:
- `unknown_comparison.pkl`
- `active_comparison.pkl`

and writes:
- `embedding_error_comparison.png`

## Thin Runner

```bash
python3 experiments/active_embedding/run_active_embedding.py --mode all
```

Modes:
- `process`
- `analysis`
- `all`
