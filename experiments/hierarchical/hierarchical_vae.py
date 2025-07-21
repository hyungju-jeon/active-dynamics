import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, LinearMapping, GaussianNoise
from actdyn.models.dynamics import MLPDynamics
from actdyn.models.model import SeqVae
from actdyn.utils.vectorfield_definition import (
    LimitCycle,
    DoubleLimitCycle,
    MultiAttractor,
)
from actdyn.utils.visualize import plot_vector_field
from actdyn.utils.rollout import Rollout, RolloutBuffer


# ----- Residual Hypernetwork -----
class ResidualHyperNet(nn.Module):
    def __init__(self, latent_dim, state_dim, hidden_dim):
        super().__init__()
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
        if latent.dim() == 2:
            latent = latent.squeeze(0)
        weights = self.hyper(latent.unsqueeze(0)).squeeze(0)
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
        f = self.base(x)
        for k in range(stage):
            f = f + self.residuals[k](x, latents[k])
        return f


# ----- VAE with Hierarchical Residual Latent Dynamics -----
class VAELatentHierarchical(nn.Module):
    def __init__(
        self, encoder, decoder, latent_dynamics, num_residuals, num_tasks, device="cpu"
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dynamics = latent_dynamics
        self.num_residuals = num_residuals
        self.num_tasks = num_tasks
        self.device = device

    def forward(self, obs_seq, task_idx, stage):
        # obs_seq: [B, T, obs_dim]
        # task_idx: int or [B]
        # stage: int
        z_samples, mu, var, _ = self.encoder(obs_seq)
        # Prepare per-task latents for each residual
        if isinstance(task_idx, int):
            latents = [
                self.latent_dynamics.latents[k][task_idx]
                for k in range(self.num_residuals)
            ]
        else:
            # batch mode: not implemented for simplicity
            raise NotImplementedError("Batch task_idx not supported in this demo.")
        # Predict next latent state using hierarchical dynamics
        z_pred = []
        for t in range(z_samples.shape[1] - 1):
            z_pred.append(self.latent_dynamics(z_samples[:, t], latents, stage=stage))
        z_pred = torch.stack(z_pred, dim=1)  # [B, T-1, latent_dim]
        # Decode
        recon = self.decoder(z_pred)
        return recon, z_samples, mu, var

    def compute_loss(self, obs_seq, task_idx, stage):
        recon, z_samples, mu, var = self.forward(obs_seq, task_idx, stage)
        # Reconstruction loss (MSE)
        target = obs_seq[:, 1:, :]
        recon_loss = F.mse_loss(recon, target)
        # KL loss (to standard normal)
        kl_loss = -0.5 * torch.sum(1 + torch.log(var) - mu**2 - var)
        return recon_loss + 1e-3 * kl_loss, recon_loss, kl_loss


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


def random_task_params(task_type):
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
    verbose = True

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

    # --- Prepare VAE model (encoder/decoder/hierarchical latent dynamics) ---
    obs_dim = state_dim  # for this synthetic data, obs = state
    encoder = MLPEncoder(input_dim=obs_dim, latent_dim=latent_dim, device=device)
    decoder = Decoder(
        LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
        GaussianNoise(output_dim=obs_dim, sigma=1.0),
        device=device,
    )
    latent_dynamics = HierarchicalMetaDynamics(
        state_dim=latent_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_residuals=num_residuals,
        num_tasks=num_tasks,
    ).to(device)
    vae = VAELatentHierarchical(
        encoder=encoder,
        decoder=decoder,
        latent_dynamics=latent_dynamics,
        num_residuals=num_residuals,
        num_tasks=num_tasks,
        device=device,
    )

    # --- Prepare training data (rollouts) ---
    rollout_buffer = RolloutBuffer(num_tasks * num_trajs)
    for i, (task_trajs, _) in enumerate(tasks):
        for traj, dtraj in task_trajs:
            rollout = Rollout()
            for t in range(traj_len - 1):
                rollout.add(
                    obs=traj[t].unsqueeze(0),
                    next_obs=traj[t + 1].unsqueeze(0),
                )
            rollout_buffer.add(rollout)

    # --- Train VAE model with hierarchical latent dynamics ---
    if verbose:
        print(
            "Training VAE (encoder/decoder/hierarchical latent dynamics) on all tasks..."
        )
    vae.train()
    optimizer = torch.optim.Adam(
        list(vae.encoder.parameters())
        + list(vae.decoder.parameters())
        + list(vae.latent_dynamics.parameters()),
        lr=1e-3,
    )
    n_epochs = 500
    stage = num_residuals  # use all residuals
    for epoch in range(n_epochs):
        total_loss = 0.0
        for task_idx in range(num_tasks):
            # For each task, sample a batch of trajectories
            batch = []
            task_rollouts = list(rollout_buffer.buffer)[
                task_idx * num_trajs : (task_idx + 1) * num_trajs
            ]
            for rollout in task_rollouts:
                obs_data = rollout._data["obs"]
                if isinstance(obs_data, list):
                    obs_seq = torch.cat(obs_data, dim=0)
                else:
                    obs_seq = obs_data
                if obs_seq.dim() == 3 and obs_seq.shape[1] == 1:
                    obs_seq = obs_seq.squeeze(1)
                batch.append(obs_seq)
            batch = torch.stack(batch, dim=0).to(device)  # [B, T, obs_dim]
            loss, recon_loss, kl_loss = vae.compute_loss(batch, task_idx, stage)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and epoch % 100 == 0:
            print(f"[VAE-Hierarchical] Epoch {epoch}, Loss: {total_loss:.4f}")

    # --- Evaluate and visualize ---
    # Pick a random task and trajectory
    task_idx = 0
    task_trajs, params = tasks[task_idx]
    traj, dtraj = task_trajs[0]
    obs_traj = traj.unsqueeze(0)  # [1, T, obs_dim]
    with torch.no_grad():
        recon, z_samples, mu, var = vae.forward(obs_traj, task_idx, stage)
    plt.figure()
    plt.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), label="True Trajectory")
    plt.plot(recon[0, :, 0].cpu(), recon[0, :, 1].cpu(), label="VAE-Hierarchical Recon")
    plt.legend()
    plt.title("VAE-Hierarchical Latent Model: Trajectory Reconstruction")
    plt.show()

    # --- Baseline: Naive model (single large MLP, no latent state) ---
    baseline_model = MLPDynamics(state_dim=state_dim, hidden_dim=256).to(device)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    if verbose:
        print("Training baseline MLP on all tasks...")
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

    # --- Visualize baseline ---
    with torch.no_grad():
        pred_baseline = baseline_model(traj)
    plt.figure()
    plt.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), label="True Trajectory")
    plt.plot(pred_baseline[:, 0].cpu(), pred_baseline[:, 1].cpu(), label="Baseline MLP")
    plt.legend()
    plt.title("Baseline MLP: Trajectory Prediction")
    plt.show()
