"""CISS RBF processing script.

Runs data generation / training for the RBF CISS experiment and writes outputs
under ``results/CISS/RBF`` by default.
"""

from __future__ import annotations

import argparse
import os

import torch

import actdyn
import actdyn.core.experiment
import actdyn.environment
import actdyn.environment.action
import actdyn.environment.observation
import actdyn.metrics
import actdyn.metrics.cost
import actdyn.metrics.uncertainty
import actdyn.models
import actdyn.models.dynamics
import actdyn.models.encoder
import actdyn.policy
import actdyn.policy.mpc
from actdyn.config import ExperimentConfig
from actdyn.utils.helper import make_uniform_sampler
from actdyn.utils.runtime import configure_runtime, ensure_dir
from actdyn.utils.visualize import set_matplotlib_style


def _default_results_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "../../results", "CISS", "RBF")


def run_processing(total_steps: int = 20000, seed: int = 0, base_dir: str | None = None) -> str:
    """Run the RBF CISS experiment and return the result directory."""
    set_matplotlib_style()
    device = configure_runtime(seed=seed)

    z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
    e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)

    base_dir = ensure_dir(_default_results_dir() if base_dir is None else base_dir)

    dz, de, du, dy = 2, 2, 2, 50
    dt = 0.1
    alpha = 5
    action_strength = 0.5
    noise_scale = 0.01

    torch.manual_seed(7)
    e = e_sampler(1)
    a, b = e.reshape(-1)

    action_model = actdyn.environment.action.IdentityActionEncoder(
        action_dim=du,
        latent_dim=dz,
        action_bounds=[-action_strength * alpha, action_strength * alpha],
        device=device,
    )

    obs_model = actdyn.environment.observation.LinearObservation(
        obs_dim=dy,
        latent_dim=dz,
        noise_scale=noise_scale,
        noise_type="gaussian",
        device=device,
    )

    duffing_env = actdyn.VectorFieldEnv(
        "duffing",
        x_range=5,
        dyn_params=torch.tensor([a, b, 0.1]),
        dt=dt,
        alpha=alpha,
        Q=noise_scale,
        action_bounds=[action_model.action_space.low, action_model.action_space.high],
        device=device,
    )
    env = actdyn.environment.EnvWrapper(duffing_env, obs_model, action_model, dt=dt, device=device)

    mapping = actdyn.models.decoder.LinearMapping(latent_dim=dz, obs_dim=dy, device=device)
    noise = actdyn.models.decoder.GaussianNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)

    dynamics = actdyn.models.dynamics.RBFDynamics(
        state_dim=dz,
        dt=env.dt,
        z_max=6,
        num_grid_pts=30,
        alpha=0.5,
        gamma=10,
        is_residual=True,
        device=device,
    )

    encoder = actdyn.models.encoder.MLPEncoder(
        obs_dim=dy,
        action_dim=du,
        latent_dim=dz,
        hidden_dim=32,
        device=device,
    )

    from actdyn.models.model import SeqStateVae

    model = SeqStateVae(
        encoder=encoder,
        dynamics=dynamics,
        decoder=decoder,
        action_encoder=action_model,
        device=device,
    )

    fisher_metric = actdyn.metrics.information.AOptimality(model=model)
    rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)
    composite_metric = actdyn.metrics.CompositeMetric([rnd_metric, fisher_metric], weights=[1.0, 1.0])

    mpc_policy = actdyn.policy.mpc.MpcICem(
        metric=composite_metric,
        model=model,
        device=device,
        horizon=10,
        num_iterations=5,
        num_samples=20,
        num_elite=10,
        chunk=5,
        verbose=False,
    )
    random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)

    exp_config = ExperimentConfig.from_yaml(
        os.path.join(os.path.dirname(__file__), "conf/RBF_video.yaml")
    )
    agent = actdyn.Agent(
        env=env,
        model=model,
        buffer_length=exp_config.training.rollout_horizon,
        policy=random_policy,
        device=device,
    )
    exp_config.results_dir = base_dir
    exp_config.training.total_steps = total_steps
    experiment = actdyn.core.experiment.Experiment(agent=agent, config=exp_config)

    decoder.set_params(obs_model)
    experiment.run()

    torch.save(
        {
            "embedding_true": e,
            "dyn_params": torch.tensor([a, b, 0.1]),
            "seed": seed,
            "total_steps": total_steps,
            "dt": dt,
            "alpha": alpha,
            "noise_scale": noise_scale,
        },
        os.path.join(base_dir, "run_metadata.pt"),
    )

    return base_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CISS RBF processing workflow")
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-dir", type=str, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_processing(total_steps=args.total_steps, seed=args.seed, base_dir=args.base_dir)


if __name__ == "__main__":
    main()
