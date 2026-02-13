# CISS Experiment Scripts

This folder keeps CISS-specific workflows split into processing and analysis entrypoints.

## RBF workflow

1. Processing (data generation / training):

```bash
python experiments/ciss/rbf_process.py --total-steps 20000
```

2. Analysis (loads saved rollouts and writes figures):

```bash
python experiments/ciss/rbf_analysis.py
```

3. Thin runner (orchestrates process + analysis):

```bash
python experiments/ciss/RBF_video.py --mode all
```

Other modes:

```bash
python experiments/ciss/RBF_video.py --mode process
python experiments/ciss/RBF_video.py --mode analysis
```

## Intro video workflow

`intro_video.py` now imports reusable animation helpers from `actdyn/utils/ciss_video.py`.
