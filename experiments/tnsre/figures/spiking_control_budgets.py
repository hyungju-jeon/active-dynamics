"""Appendix grouped bars of saved warm-start control success at each budget.

Run from the repository root:
    .venv/bin/python -m experiments.tnsre.figures.spiking_control_budgets

Uses the original manuscript experiment, not the later evidence-drive variants.
Recomputes means and SEM from session outcomes and checks the saved summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .assets import (
    _apply_asset_style, _asset_baseline_policy_color, _asset_policy_label,
    save_figure,
)
from .theme import style_experiment_axis

from .spiking_sessions import BUDGET_PA2_S

WIDTH = 516 / 72.27
POLICIES = ("adaptive", "active_myopic", "flex_rollback", "rhc", "prbs", "random")
REFERENCES = ("reduced_fit", "spread", "front")
METHODS = POLICIES + REFERENCES
REF_LABELS = dict(reduced_fit="Fitted", spread="Uniform", front="Full")
REF_COLORS = dict(reduced_fit="#A0A0A0", spread="#C4C4C4", front="#D6C5AE")
EXTENSION = Path("results/tnsre/20260929_spiking_control_budgets/extension")


def load_scores(eval_dir: Path, extension_dir: Path = EXTENSION) -> tuple[pd.DataFrame, dict]:
    """Check all plotted cells against raw outcomes; record input hashes."""
    protocol_path = eval_dir / "protocol.json"
    protocol = json.loads(protocol_path.read_text())["protocol"]
    summary_path = eval_dir / "task_summary_warm.csv"
    summary = pd.read_csv(summary_path)
    warm_paths = sorted((eval_dir / "task_warm").glob("worker_*.csv"))
    reference_paths = sorted((eval_dir / "task").glob("worker_*.csv"))
    warm = pd.concat([pd.read_csv(p) for p in warm_paths], ignore_index=True)
    fresh = pd.concat([pd.read_csv(p) for p in reference_paths], ignore_index=True)
    # Model-free inputs do not plan: the scorer shares these original outcomes.
    raw = pd.concat([warm, fresh[fresh.controller.isin(("spread", "front"))]])
    extension_summary = extension_dir / "task_summary_warm.csv"
    extension_paths = sorted((extension_dir / "task_warm").glob("worker_*.csv"))
    summary = pd.concat([summary, pd.read_csv(extension_summary)], ignore_index=True)
    raw = pd.concat([raw, *[pd.read_csv(p) for p in extension_paths]], ignore_index=True)
    protocol["budgets"]["overturn"] = [5.0, 10.0, 12.0, 15.0]
    selected = []
    excluded = {}
    for task in ("overturn", "force"):
        expected_tasks = set(range(protocol["task_seeds"][task],
                                   protocol["task_seeds"][task] + protocol["task_sessions"]))
        shared_valid = None
        for budget in protocol["budgets"][task]:
            for policy in METHODS:
                checkpoint = 20 if policy in POLICIES else -1
                rows = raw[(raw.policy == policy) & (raw.checkpoint == checkpoint)
                           & (raw.task == task) & (raw.budget == budget)]
                expected_seeds = set(range(20)) if policy not in ("spread", "front") else {-1}
                assert set(rows.seed) == expected_seeds, (task, budget, policy, "seeds")
                assert not rows.duplicated(["seed", "task_seed"]).any()
                for _, run in rows.groupby("seed"):
                    assert set(run.task_seed) == expected_tasks
                    valid_ids = set(run.loc[run.valid == 1, "task_seed"])
                    if shared_valid is None:
                        shared_valid = valid_ids
                    assert valid_ids == shared_valid, "Task exclusions differ between methods"
                valid = rows[rows.valid == 1]
                seed_means = valid.groupby("seed").success.mean()
                mean = float(seed_means.mean())
                sem = float(seed_means.sem()) if len(seed_means) > 1 else 0.0
                saved = summary[(summary.policy == policy) & (summary.checkpoint == checkpoint)
                                & (summary.task == task) & (summary.budget == budget)]
                assert len(saved) == 1
                saved = saved.iloc[0]
                np.testing.assert_allclose([mean, sem], [saved.success, saved.sem_over_seeds],
                                           rtol=1e-12, atol=1e-14)
                assert (len(seed_means), len(valid)) == (saved.n_seeds, saved.n_sessions)
                selected.append(dict(task=task, budget=budget, policy=policy, success=mean,
                                     sem=sem, n_seeds=len(seed_means), n_sessions=len(valid)))
        excluded[task] = sorted(expected_tasks - shared_valid)
    paths = [protocol_path, summary_path, extension_summary, *warm_paths, *reference_paths, *extension_paths,
             *[extension_dir / name for name in ("protocol.json", "source_hashes.json",
                                                 "model_input_hashes.json", "pilot_validation.json")]]
    provenance = dict(
        checkpoint=20, controller="warm-start iCEM", budgets=protocol["budgets"],
        excluded_task_seeds=excluded, validated_cells=len(selected),
        uncertainty="SEM of within-seed success rates; model-free references have no seed SEM",
        fitted_reference="One fixed fitted model evaluated with 20 seed-specific observation maps",
        ordering="Methods descending by mean success at the highest budget per task; increasing budgets within method",
        inputs={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    )
    return pd.DataFrame(selected), provenance


def make_figure(scores: pd.DataFrame):
    """Group budgets by method; rank methods at each task's highest budget."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _apply_asset_style(plt)
    from matplotlib.patches import Patch
    from matplotlib.colors import to_rgb
    fig, ax = plt.subplots(figsize=(WIDTH, 2.25))
    fig.subplots_adjust(left=0.075, right=0.986, bottom=0.19, top=0.83)
    for ax, task in [(ax, "overturn")]:
        budgets = sorted(scores.loc[scores.task == task, "budget"].unique())
        ranking = scores[(scores.task == task) & (scores.budget == budgets[-1])].set_index("policy")
        order = sorted(METHODS, key=lambda p: (-ranking.loc[p, "success"], METHODS.index(p)))
        x = np.arange(len(order))
        width = 0.78 / len(budgets)
        strengths = np.linspace(0.4, 1.0, len(budgets))
        for j, policy in enumerate(order):
            rows = scores[(scores.task == task) & (scores.policy == policy)].set_index("budget").loc[budgets]
            color = _asset_baseline_policy_color(policy) if policy in POLICIES else REF_COLORS[policy]
            positions = j + (np.arange(len(budgets)) - (len(budgets) - 1) / 2) * width
            colors = [tuple(1 - strength * (1 - c) for c in to_rgb(color)) for strength in strengths]
            ax.bar(positions, rows.success, width=width * 0.91,
                   color=colors, edgecolor="#333333", linewidth=0.35, zorder=3)
            if policy not in ("spread", "front"):
                ax.errorbar(positions, rows.success, yerr=rows["sem"],
                            fmt="none", ecolor="#252525", elinewidth=0.55,
                            capsize=1.0, capthick=0.55, zorder=4)
        ax.set_xticks(x, [_asset_policy_label(p) if p in POLICIES else REF_LABELS[p] for p in order])
        ax.set_xlim(-0.55, len(order) - 0.45)
        ax.set_ylim(0, 1.04)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlabel("Method", labelpad=3)
        ax.set_ylabel("Success rate", labelpad=3)
        style_experiment_axis(ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linewidth=0.4, color="#E4E4E4")
        handles = [Patch(facecolor=str(1 - 0.55 * strength), edgecolor="#333333", linewidth=0.35,
                         label=rf"{BUDGET_PA2_S*b:g} pA$^2$ s")
                   for b, strength in zip(budgets, strengths)]
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.03),
                  ncol=len(budgets), frameon=False, handlelength=1.7, columnspacing=1.25)
    return fig, plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path,
                        default=Path("results/tnsre/20260924_snn_sessions_m2"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/tnsre/20260929_spiking_control_budgets"))
    parser.add_argument("--extension-dir", type=Path, default=EXTENSION)
    args = parser.parse_args()
    scores, provenance = load_scores(args.experiment_dir / "eval", args.extension_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = scores[scores.task == "overturn"].copy()
    scores.to_csv(args.output_dir / "plotted_scores.csv", index=False)
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    fig, plt = make_figure(scores)
    output = args.output_dir / "appendix_spiking_control_budgets.pdf"
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches=None)
    save_figure(fig, output, plt_module=plt)
    print(f"Validated {len(scores)} bars against raw session outcomes; wrote {output}")


if __name__ == "__main__":
    main()
