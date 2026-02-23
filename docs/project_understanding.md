# Active Dynamics — Project Understanding (from `docs/` + `TBME.pdf` + code structure)

## 1) Core problem the project is solving
This project targets **online system identification for nonlinear, partially observed dynamical systems** under real-time constraints.

In plain terms:
- The system has hidden state and unknown dynamics parameters.
- Passive observation is often not enough (especially with attractors/metastable dynamics), because trajectories do not naturally visit informative regions.
- So the controller should not only do task control, but also **actively choose inputs that make future observations more informative** for identifying the dynamics.

This is framed as a dual objective:
1. estimate latent state and parameters online,
2. select controls that maximize expected information gain (or strong surrogates of it).

---

## 2) Scope implied by the paper draft (`TBME.pdf`)
From `docs/active-dynamics-writing/TBME.pdf`, the paper direction is:
- **Domain motivation:** neural dynamics / latent neural state models with practical experimental constraints.
- **Inference stack:**
  - EKF-style latent filtering (local Gaussian approximation),
  - Laplace-style online parameter posterior update.
- **Planning stack:** sampling-based MPC (iCEM-style) that scores candidate control sequences by information criteria (Fisher/A-/D-optimality-like objectives), aiming for bounded per-step compute.

So the intended contribution is an **information-guided online planner** coupled to approximate Bayesian filtering/learning for latent nonlinear dynamics.

---

## 3) What the docs reveal about research direction (beyond current TBME draft)
The docs suggest this is broader than one fixed method and currently evolving along two advanced tracks:

### A) Identifiability-Aware Active Filtering (IAAF)
(`docs/idea2_iaaf_identifiability_aware_active_filtering.md`, `docs/paper_methods_rigorous.md`)
- Motivation: local Fisher curvature can be misleading under nonlinear non-identifiability / multimodal posteriors.
- Proposal: plan using objectives that emphasize
  - posterior contraction,
  - belief-weighted information geometry,
  - separation of currently confusable parameter hypotheses.

### B) Amortized EIG-MPC
(`docs/idea4_amortized_eig_mpc_nonlinear_ssm.md`, `docs/paper_draft_iaaf_amortized_eig_mpc.md`)
- Motivation: exact expected information gain (EIG) is too expensive inside MPC.
- Proposal: learn a cheap surrogate (BA/InfoNCE-type MI lower bounds) so MPC can score information gain quickly.

Interpretation: the project is transitioning from **classical local Fisher active design** toward **more robust, scalable information-theoretic planning** for realistic nonlinear settings.

---

## 4) How this maps to the codebase today
From code structure already present in this repo, the software is a modular active-learning framework:
- `actdyn/core/*`: agent + experiment loop.
- `actdyn/models/*`: latent models (`SeqVae`, filtering embedding, ensembles).
- `actdyn/policy/*`: MPC (iCEM), random, off-policy.
- `actdyn/metrics/*`: information metrics (Fisher/A-opt/D-opt, embedding-fisher) and task metrics.
- `actdyn/environment/*`: benchmark environments.
- `experiments/*`: runnable tracks and analysis scripts.

This matches the research scope: **belief/model + metric + planner** are composable, enabling comparison of information objectives under a unified runtime.

---

## 5) My distilled understanding of the project
`active-dynamics` is a **research platform for active identification of latent dynamical systems**, with special emphasis on:
- real-time online inference,
- information-aware control input design,
- practical experiment pipelines (run/sweep/analyze),
- and method evolution from local Fisher proxies toward richer identifiability/EIG objectives.

So the project is not just “build a dynamics model.”
It is about **closing the loop between what the model believes and what the controller chooses to observe next**, to learn dynamics faster and more reliably in hard nonlinear regimes.

---

## 6) Current conceptual gaps to resolve (for paper + implementation alignment)
Based on docs/manuscript mix, key alignment tasks likely needed:
1. **Single canonical method story**: Fisher-only TBME narrative vs newer IAAF/amortized-EIG narrative should be unified.
2. **Claim-to-code mapping**: each mathematical objective should map to a concrete metric class + experiment config.
3. **Evaluation protocol consistency**: equal compute-budget comparisons and clear calibration/identifiability metrics should be standardized across tracks.
4. **Approximation transparency**: explicitly state where Gaussian/local assumptions are used and where they can fail.

These are normal for an active research codebase and do not contradict the core direction.
