# %%
import matplotlib.pyplot as plt
import numpy as np

from experiments.analyze_results import analyze_all_models

if __name__ == "__main__":
    # Compare Least square from true trajectory and plug in KL computation

    # Things we can do to prioritize learning of dynamics
    # 1. Schedule
    # 2. Multistep (Beware that KL term is no longer gaussian - need some approximation)
    # 2-1. Check dreamer implementation
    # 3. Debug dynamics model (without any observation)

    # Isabel
    # 1. Change dt
    # 2. Rendering
    # 3. Fully observable with inflated state space

    #

    from pathlib import Path

    exp_folder = Path("/home/hyungju/Desktop/active-dynamics/results/offline_debug/off_policy")

    offline_data = analyze_all_models(exp_folder, is_offline=True)

    plt.figure(figsize=(10, 6))
    for k, v in offline_data.items():
        y = v["kl_divergence"][0]
        plt.axvline(x=len(y), color="gray", linestyle="--", lw=1)
        y = np.pad(y, (0, max(0, 2000 - len(y))), mode="edge")
        plt.plot(y, label=f"Offline KL - {k}", alpha=0.75, lw=1.5)

    # plt.show()

    # plt.figure(figsize=(10, 6))
    # Analyze all models
    online_data = analyze_all_models(exp_folder)
    for k, v in online_data.items():
        if "lr_0.01_" in k:
            y = v["kl_divergence"][0]
            plt.plot(y, label=f"Online KL - {k}", alpha=0.75, lw=2, color="k")
            # break
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.ylim(-0.0, 3)
    plt.xlim(0, 2000)
    plt.xlabel("Training Steps")
    plt.ylabel("KL Divergence")
    plt.title(f"Off-/Online Training Comparison (KL)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

    # ------------
    plt.figure(figsize=(10, 6))
    for k, v in offline_data.items():
        y = v["log_like"][0]
        plt.axvline(x=len(y), color="gray", linestyle="--", lw=1)
        y = np.pad(y, (0, max(0, 5000 - len(y))), mode="edge")
        plt.plot(y, label=f"Offline LL - {k}", alpha=0.75, lw=1.5)

    # plt.show()

    # plt.figure(figsize=(10, 6))
    # Analyze all models
    online_data = analyze_all_models(exp_folder)
    for k, v in online_data.items():
        if "lr_0.01_" in k:
            y = v["log_like"][0]
            plt.plot(y, label=f"Online LL - {k}", alpha=0.75, lw=2, color="k")
            break
    plt.grid(True, linestyle="--", alpha=0.5)

    # plt.ylim(-0.0, 3)
    plt.xlim(0, 5000)
    plt.xlabel("Training Steps")
    plt.ylabel("Log Likelihood")
    plt.title(f"Off-/Online Training Comparison (LL)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
