import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from typing import cast
import random

from actdyn.models.encoder import RNNEncoder
from actdyn.models.dynamics import MLPDynamics
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.utils.vectorfield_definition import (
    LimitCycle,
    DoubleLimitCycle,
    MultiAttractor,
)
from actdyn.utils.visualize import plot_vector_field


# ----- Residual Hypernetwork -----
class ResidualHyperNet(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dim):
        super().__init__()
        # Hypernetwork: latent -> weights for residual MLP
        self.hyper = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(
                32,
                (state_dim) * hidden_dim
                + hidden_dim * state_dim
                + hidden_dim
                + state_dim,
            ),
        )
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, latent):
        # latent: [latent_dim] or [1, latent_dim]
        if latent.dim() == 2:
            latent = latent.squeeze(0)
        # Now latent is [latent_dim]
        weights = self.hyper(latent.unsqueeze(0)).squeeze(0)  # [total_params]
        batch = x.shape[0]
        idx = 0
        in_dim = self.state_dim
        h_dim = self.hidden_dim
        out_dim = self.state_dim
        w1 = weights[idx : idx + in_dim * h_dim].reshape(h_dim, in_dim)
        idx += in_dim * h_dim
        b1 = weights[idx : idx + h_dim].reshape(h_dim)
        idx += h_dim
        w2 = weights[idx : idx + h_dim * out_dim].reshape(out_dim, h_dim)
        idx += h_dim * out_dim
        b2 = weights[idx : idx + out_dim].reshape(out_dim)
        # Apply to all x in the batch
        h = F.linear(x, w1, b1)
        h = F.relu(h)
        out = F.linear(h, w2, b2)
        return out


# ----- Hierarchical Meta-Dynamics Model (staged) -----
class HierarchicalMetaDynamics(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dim, num_residuals, num_tasks):
        super().__init__()
        self.base = MLPDynamics(state_dim, hidden_dim)
        self.residuals = nn.ModuleList(
            [
                ResidualHyperNet(latent_dim, state_dim, hidden_dim)
                for _ in range(num_residuals)
            ]
        )
        # Per-stage, per-task latents: [stage][task]
        self.latents = nn.ModuleList(
            [
                nn.ParameterList(
                    [nn.Parameter(torch.zeros(latent_dim)) for _ in range(num_tasks)]
                )
                for _ in range(num_residuals)
            ]
        )
        self.state_dim = state_dim
        self.num_residuals = num_residuals
        self.num_tasks = num_tasks

    def forward(self, x, latents, stage):
        # latents: list of [latent_dim] (one per residual, per task)
        # stage: int, how many residuals to use (0 = only base)
        f = self.base(x)
        for k in range(stage):
            f = f + self.residuals[k](x, latents[k])
        return f


# --- Data Generation: Multiple VectorField Tasks ----
def generate_task_data(
    num_tasks,
    num_trajs,
    traj_len,
    state_dim=2,
    device="cpu",
    task_types=None,
    task_params=None,
):
    tasks = []
    if task_types is None:
        task_types = ["limitcycle"] * num_tasks
    if task_params is None:
        task_params = [{} for _ in range(num_tasks)]
    for i in range(num_tasks):
        ttype = task_types[i].lower()
        params = dict(task_params[i])  # copy to avoid mutating input
        params["type"] = ttype  # store type explicitly
        if ttype == "limitcycle":
            w = params.get("w", -1.0 + 0.5 * i)
            d = params.get("d", 1.0 + 0.1 * i)
            vf = LimitCycle(w=w, d=d)
        elif ttype == "doublelimitcycle":
            w = params.get("w", -1.0 + 0.5 * i)
            d = params.get("d", 1.0 + 0.1 * i)
            vf = DoubleLimitCycle(w=w, d=d)
        elif ttype == "multiattractor":
            w_attractor = params.get("w_attractor", 0.1)
            length_scale = params.get("length_scale", 0.5)
            alpha = params.get("alpha", 0.1)
            vf = MultiAttractor(
                w_attractor=w_attractor, length_scale=length_scale, alpha=alpha
            )
        else:
            raise ValueError(f"Unknown vector field type: {ttype}")
        task_trajs = []
        for _ in range(num_trajs):
            x0 = torch.randn(state_dim, device=device)
            traj = [x0]
            for t in range(traj_len - 1):
                dx = vf(traj[-1].unsqueeze(0)).squeeze(0)
                x_next = traj[-1] + 0.1 * dx  # simple Euler step
                traj.append(x_next)
            traj = torch.stack(traj, dim=0)  # [traj_len, state_dim]
            dtraj = vf(traj)  # [traj_len, state_dim]
            task_trajs.append((traj, dtraj))
        tasks.append((task_trajs, params))
    return tasks


def visualize_results(
    model,
    tasks,
    vis_idx,
    device,
    x_range=2.5,
    n_grid=50,
    task_params=None,
):
    import matplotlib.pyplot as plt
    from typing import cast
    from actdyn.utils.visualize import plot_vector_field
    from actdyn.utils.vectorfield_definition import (
        LimitCycle,
        DoubleLimitCycle,
        MultiAttractor,
    )

    # 2x5 subplot: true, base, res1, res2, combined
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    for row, task_idx in enumerate(vis_idx):
        # Use the actual task type and params from task_params
        if task_params is not None:
            params = dict(task_params[task_idx])
            ttype = params.get("type", "limitcycle")
        else:
            _, params = tasks[task_idx]
            ttype = params.get("type", "limitcycle")
        if ttype == "limitcycle":
            if "w" not in params or "d" not in params:
                raise ValueError(
                    f"LimitCycle requires 'w' and 'd' in params, got {params}"
                )
            vf = LimitCycle(w=params["w"], d=params["d"])
        elif ttype == "doublelimitcycle":
            if "w" not in params or "d" not in params:
                raise ValueError(
                    f"DoubleLimitCycle requires 'w' and 'd' in params, got {params}"
                )
            vf = DoubleLimitCycle(w=params["w"], d=params["d"])
        elif ttype == "multiattractor":
            vf = MultiAttractor(
                w_attractor=params.get("w_attractor", 0.1),
                length_scale=params.get("length_scale", 0.5),
                alpha=params.get("alpha", 0.1),
            )
        else:
            raise ValueError(f"Unknown vector field type: {ttype}")

        def base_dyn(x):
            return model.base(x)

        latents_res1 = [cast(nn.ParameterList, model.latents[0])[task_idx]]

        def res1_dyn(x):
            return model.residuals[0](x, latents_res1[0])

        latents_res2 = [cast(nn.ParameterList, model.latents[1])[task_idx]]

        def res2_dyn(x):
            return model.residuals[1](x, latents_res2[0])

        def combined_dyn(x):
            return (
                model.base(x)
                + model.residuals[0](x, latents_res1[0])
                + model.residuals[1](x, latents_res2[0])
            )

        plot_vector_field(
            lambda x: vf(x),
            ax=axes[row, 0],
            x_range=x_range,
            n_grid=n_grid,
            device=device,
        )
        axes[row, 0].set_title(f"Task {task_idx+1}: True")
        plot_vector_field(
            base_dyn, ax=axes[row, 1], x_range=x_range, n_grid=n_grid, device=device
        )
        axes[row, 1].set_title(f"Task {task_idx+1}: Base")
        plot_vector_field(
            res1_dyn, ax=axes[row, 2], x_range=x_range, n_grid=n_grid, device=device
        )
        axes[row, 2].set_title(f"Task {task_idx+1}: Res_1")
        plot_vector_field(
            res2_dyn, ax=axes[row, 3], x_range=x_range, n_grid=n_grid, device=device
        )
        axes[row, 3].set_title(f"Task {task_idx+1}: Res_2")
        plot_vector_field(
            combined_dyn, ax=axes[row, 4], x_range=x_range, n_grid=n_grid, device=device
        )
        axes[row, 4].set_title(f"Task {task_idx+1}: Combined")
    plt.tight_layout()
    plt.show()


def visualize_latent_recovery(
    model,
    x,
    dx,
    trained_latents,
    recovered_latents,
    num_residuals,
    device,
    x_range=2.5,
    n_grid=50,
):
    import matplotlib.pyplot as plt
    from actdyn.utils.visualize import plot_vector_field

    def trained_dyn(x_):
        return (
            model.base(x_)
            + model.residuals[0](x_, trained_latents[0])
            + model.residuals[1](x_, trained_latents[1])
        )

    def inferred_dyn(x_):
        return (
            model.base(x_)
            + model.residuals[0](x_, recovered_latents[0])
            + model.residuals[1](x_, recovered_latents[1])
        )

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    plot_vector_field(
        trained_dyn, ax=axes2[0], x_range=x_range, n_grid=n_grid, device=device
    )
    axes2[0].set_title("Trained Latent Code")
    plot_vector_field(
        inferred_dyn, ax=axes2[1], x_range=x_range, n_grid=n_grid, device=device
    )
    axes2[1].set_title("Inferred Latent Code (Online)")
    plt.tight_layout()
    plt.show()

    # Visualization for baseline


def visualize_baseline_new_task(
    test_task_type,
    test_task_param,
    baseline_adapt,
    x_test,
    dx_test,
    device,
    x_range=2.5,
    n_grid=50,
):
    import matplotlib.pyplot as plt
    from actdyn.utils.visualize import plot_vector_field
    from actdyn.utils.vectorfield_definition import LimitCycle, DoubleLimitCycle

    if test_task_type == "limitcycle":
        vf = LimitCycle(w=test_task_param["w"], d=test_task_param["d"])
    elif test_task_type == "doublelimitcycle":
        vf = DoubleLimitCycle(w=test_task_param["w"], d=test_task_param["d"])
    else:
        raise ValueError(f"Unknown test task type: {test_task_type}")

    def model_dyn(x):
        return baseline_adapt(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_vector_field(
        lambda x: vf(x), ax=axes[0], x_range=x_range, n_grid=n_grid, device=device
    )
    axes[0].set_title("Unseen Task: True Vector Field")
    plot_vector_field(
        model_dyn, ax=axes[1], x_range=x_range, n_grid=n_grid, device=device
    )
    axes[1].set_title("Baseline MLP after Adaptation")
    plt.tight_layout()
    plt.show()


def visualize_new_task(
    test_task_type,
    test_task_param,
    model,
    x_test,
    dx_test,
    recovered_latents_test,
    device,
    x_range=2.5,
    n_grid=50,
):
    import matplotlib.pyplot as plt
    from actdyn.utils.visualize import plot_vector_field
    from actdyn.utils.vectorfield_definition import LimitCycle, DoubleLimitCycle

    if test_task_type == "limitcycle":
        vf = LimitCycle(w=test_task_param["w"], d=test_task_param["d"])
    elif test_task_type == "doublelimitcycle":
        vf = DoubleLimitCycle(w=test_task_param["w"], d=test_task_param["d"])
    else:
        raise ValueError(f"Unknown test task type: {test_task_type}")

    def model_dyn(x):
        return model(x, recovered_latents_test, stage=num_residuals)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_vector_field(
        lambda x: vf(x), ax=axes[0], x_range=x_range, n_grid=n_grid, device=device
    )
    axes[0].set_title("Unseen Task: True Vector Field")
    plot_vector_field(
        model_dyn, ax=axes[1], x_range=x_range, n_grid=n_grid, device=device
    )
    axes[1].set_title("Model Prediction after Adaptation")
    plt.tight_layout()
    plt.show()


def infer_latent_code(
    model, x, dx, num_residuals, latent_dim, device, num_steps=1, lr=1e-2
):
    """
    Online adaptation: infer a latent code for a new task by optimizing it on (x, dx) with model weights frozen.
    Args:
        model: trained HierarchicalMetaDynamics
        x: [N, state_dim] tensor of states (or state-action if using actions)
        dx: [N, state_dim] tensor of targets
        num_residuals: number of residuals in the model
        latent_dim: dimension of latent code
        device: torch device
        num_steps: number of optimization steps
        lr: learning rate for latent code
    Returns:
        latents_per_residual: list of optimized latent codes (one per residual)
    Usage:
        latents = infer_latent_code(model, x, dx, num_residuals, latent_dim, device)
        pred = model(x, latents, stage=num_residuals)
    """
    # Initialize a latent code for each residual
    latents = [
        torch.zeros(latent_dim, requires_grad=True, device=device)
        for _ in range(num_residuals)
    ]
    optimizer = torch.optim.AdamW(latents, lr=lr)
    model.eval()
    traj_len = x.shape[0]
    for step in range(traj_len - 20):
        pred = model(x[step : step + 20], latents, stage=num_residuals)
        loss = F.mse_loss(pred, dx[step : step + 20])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Detach and return
    return [l.detach() for l in latents]


def orthogonality_regularizer(model, latents_per_residual):
    """
    Computes orthogonality penalty for the first-layer weights of all residuals for a given task,
    and also between each residual and the base network.
    Args:
        model: HierarchicalMetaDynamics
        latents_per_residual: list of latent codes for each residual (for a single task)
    Returns:
        penalty: scalar tensor
    """
    first_layer_weights = []
    for k in range(len(latents_per_residual)):
        res = model.residuals[k]
        weights = res.hyper(latents_per_residual[k].unsqueeze(0)).squeeze(0)
        in_dim = model.state_dim
        h_dim = res.hidden_dim
        w1 = weights[: in_dim * h_dim].reshape(h_dim, in_dim)
        first_layer_weights.append(w1.flatten())
    # Get base network's first-layer weights
    base_w1 = model.base.network[0].weight.flatten()
    penalty = 0.0
    # Residual-residual orthogonality
    for i in range(len(first_layer_weights)):
        for j in range(i + 1, len(first_layer_weights)):
            dot = torch.dot(first_layer_weights[i], first_layer_weights[j])
            penalty = penalty + dot**2
    # Residual-base orthogonality
    for i in range(len(first_layer_weights)):
        dot_base = torch.dot(first_layer_weights[i], base_w1)
        penalty = penalty + dot_base**2
    return penalty


def random_task_params(task_type):
    # Make tasks more diverse by sampling from a wide range
    if task_type == "limitcycle":
        w = random.uniform(-4.0, 4.0)
        d = random.uniform(0.5, 3.0)
        return {"w": w, "d": d}
    elif task_type == "doublelimitcycle":
        w = random.uniform(-4.0, 4.0)
        d = random.uniform(0.5, 3.0)
        return {"w": w, "d": d}
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# %%
if __name__ == "__main__":
    torch.manual_seed(0)
    random.seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_dim = 2
    latent_dim = 4
    hidden_dim = 32
    num_residuals = 2
    num_tasks = 8
    num_trajs = 100
    traj_len = 200
    ortho_lambda = 1e-4  # strength of orthogonality regularizer
    verbose = True  # set to False to suppress print logs

    # --- Randomly generate diverse tasks ---
    task_types = [
        random.choice(["limitcycle", "doublelimitcycle"]) for _ in range(num_tasks)
    ]
    task_params = [random_task_params(ttype) for ttype in task_types]
    tasks = generate_task_data(
        num_tasks=num_tasks,
        num_trajs=num_trajs,
        traj_len=traj_len,
        state_dim=state_dim,
        device=device,
        task_types=task_types,
        task_params=task_params,
    )
    model = HierarchicalMetaDynamics(
        state_dim, latent_dim, hidden_dim, num_residuals, num_tasks
    ).to(device)
    # Stage 0: Train base dynamics only
    for p in model.base.parameters():
        p.requires_grad = True
    for k in range(num_residuals):
        for p in model.residuals[k].parameters():
            p.requires_grad = False
        for p in model.latents[k].parameters():
            p.requires_grad = False
    optimizer = torch.optim.Adam(model.base.parameters(), lr=1e-3)
    if verbose:
        print("Stage 0: Training base dynamics f_0 only")
    for epoch in range(500):
        total_loss = torch.tensor(0.0, device=device)
        for i, (task_trajs, _) in enumerate(tasks):
            x = torch.cat([traj for traj, _ in task_trajs], dim=0)
            dx = torch.cat([dtraj for _, dtraj in task_trajs], dim=0)
            latents_per_residual = [
                torch.zeros(latent_dim, device=device) for _ in range(num_residuals)
            ]
            pred = model(x, latents_per_residual, stage=0)
            loss = F.mse_loss(pred, dx)
            total_loss = total_loss + loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"[Base] Epoch {epoch}, Loss: {total_loss.item():.4f}")
    # Progressive stages: add one residual at a time
    for stage in range(1, num_residuals + 1):
        if verbose:
            print(f"Stage {stage}: Training residual {stage-1} and per-task latents")
        for k in range(num_residuals):
            for p in model.residuals[k].parameters():
                p.requires_grad = k == stage - 1
            for p in model.latents[k].parameters():
                p.requires_grad = k == stage - 1
        for p in model.base.parameters():
            p.requires_grad = False
        params = list(model.residuals[stage - 1].parameters()) + list(
            model.latents[stage - 1].parameters()
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)
        for epoch in range(500):
            total_loss = torch.tensor(0.0, device=device)
            for i, (task_trajs, _) in enumerate(tasks):
                x = torch.cat([traj for traj, _ in task_trajs], dim=0)
                dx = torch.cat([dtraj for _, dtraj in task_trajs], dim=0)
                latents_per_residual = []
                for k in range(num_residuals):
                    if k < stage:
                        from typing import cast

                        latents_per_residual.append(
                            cast(nn.ParameterList, model.latents[k])[i]
                        )
                    else:
                        latents_per_residual.append(
                            torch.zeros(latent_dim, device=device)
                        )
                pred = model(x, latents_per_residual, stage=stage)
                loss = F.mse_loss(pred, dx)
                ortho_penalty = orthogonality_regularizer(
                    model, latents_per_residual[:stage]
                )
                loss = loss + ortho_lambda * ortho_penalty
                total_loss = total_loss + loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if verbose and epoch % 100 == 0:
                ortho_val = float(ortho_penalty)
                print(
                    f"[Residual {stage-1}] Epoch {epoch}, Loss: {total_loss.item():.4f}, Ortho: {ortho_val:.6f}"
                )
    # %%
    # --- Test latent recovery on a training task ---
    if verbose:
        print("\nTesting latent code recovery on a training task...")
    task_idx = 0  # Try the first training task
    task_trajs, _ = tasks[task_idx]
    x = torch.cat([traj for traj, _ in task_trajs], dim=0)
    dx = torch.cat([dtraj for _, dtraj in task_trajs], dim=0)
    recovered_latents = infer_latent_code(
        model, x, dx, num_residuals, latent_dim, device, num_steps=200, lr=1e-2
    )
    from typing import cast

    trained_latents = [
        cast(nn.ParameterList, model.latents[k])[task_idx].detach()
        for k in range(num_residuals)
    ]
    for k in range(num_residuals):
        mse = F.mse_loss(recovered_latents[k], trained_latents[k]).item()
        if verbose:
            print(
                f"Residual {k+1}: MSE between recovered and trained latent: {mse:.6f}"
            )
    pred_recovered = model(x, recovered_latents, stage=num_residuals)
    pred_trained = model(x, trained_latents, stage=num_residuals)
    mse_pred = F.mse_loss(pred_recovered, dx).item()
    mse_trained = F.mse_loss(pred_trained, dx).item()
    if verbose:
        print(f"Prediction MSE with recovered latents: {mse_pred:.6f}")
        print(f"Prediction MSE with trained latents: {mse_trained:.6f}")
    # Visualizations
    visualize_results(model, tasks, [0, 3], device)
    visualize_latent_recovery(
        model, x, dx, trained_latents, recovered_latents, num_residuals, device
    )
    # %%
    # --- Try on a newly seen (unseen) task ---
    # Generate a random test task not in the training set
    test_task_type = random.choice(["limitcycle", "doublelimitcycle"])
    test_task_param = random_task_params(test_task_type)
    test_tasks = generate_task_data(
        num_tasks=1,
        num_trajs=1,
        traj_len=1000,
        state_dim=state_dim,
        device=device,
        task_types=[test_task_type],
        task_params=[test_task_param],
    )
    test_trajs, _ = test_tasks[0]
    # Use only a single trajectory for online adaptation
    single_traj, single_dtraj = test_trajs[0]  # or random.choice(test_trajs)
    x_test = single_traj
    dx_test = single_dtraj
    # Online adaptation: infer latent codes for the new task
    recovered_latents_test = infer_latent_code(
        model,
        x_test,
        dx_test,
        num_residuals,
        latent_dim,
        device,
        num_steps=10,
        lr=1e-2,
    )
    # Model prediction after adaptation
    pred_test = model(x_test, recovered_latents_test, stage=num_residuals)
    mse_test = F.mse_loss(pred_test, dx_test).item()
    print(f"\n[Generalization] Prediction MSE on unseen test task: {mse_test:.6f}")

    # Visualization for the new task
    visualize_new_task(
        test_task_type,
        test_task_param,
        model,
        x_test,
        dx_test,
        recovered_latents_test,
        device,
    )
    # %%

    # --- Baseline: Naive model (single large MLP, no residuals) ---
    # Create baseline model
    baseline_model = MLPDynamics(state_dim, 256).to(device)
    # Train baseline model on all tasks (no residuals, no latents)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    for epoch in range(500):
        total_loss = torch.tensor(0.0, device=device)
        for i, (task_trajs, _) in enumerate(tasks):
            x = torch.cat([traj for traj, _ in task_trajs], dim=0)
            dx = torch.cat([dtraj for _, dtraj in task_trajs], dim=0)
            pred = baseline_model(x)
            loss = F.mse_loss(pred, dx)
            total_loss = total_loss + loss
        baseline_optimizer.zero_grad()
        total_loss.backward()
        baseline_optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"[Baseline] Epoch {epoch}, Loss: {total_loss.item():.4f}")

    # %%
    # Online adaptation for baseline: optimize all weights on the unseen task (like fine-tuning)
    baseline_model.eval()
    baseline_adapt = MLPDynamics(state_dim, 256).to(device)
    baseline_adapt.load_state_dict(baseline_model.state_dict())
    adapt_optimizer = torch.optim.AdamW(baseline_adapt.parameters(), lr=1e-2)

    traj_len = x_test.shape[0]
    for step in range(traj_len - 20):
        pred = baseline_adapt(x_test[step : step + 20])
        loss = F.mse_loss(pred, dx_test[step : step + 20])
        adapt_optimizer.zero_grad()
        loss.backward()
        adapt_optimizer.step()

    pred_baseline = baseline_adapt(x_test)
    mse_baseline = F.mse_loss(pred_baseline, dx_test).item()
    print(
        f"[Generalization] Baseline MLP Prediction MSE on unseen test task: {mse_baseline:.6f}"
    )

    visualize_baseline_new_task(
        test_task_type, test_task_param, baseline_adapt, x_test, dx_test, device
    )
    print(
        f"\n[Comparison] Hierarchical MSE: {mse_test:.6f}, Baseline MLP MSE: {mse_baseline:.6f}"
    )
