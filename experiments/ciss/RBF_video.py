
def main() -> None:
    # %%
    import os
    
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from einops import einsum, rearrange, repeat
    from torch.nn.functional import softplus
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.tensorboard.writer import SummaryWriter
    from tqdm import tqdm
    
    # from external.integrative_inference.src.utils import save_model, load_model
    import actdyn
    import actdyn.core
    import actdyn.core.experiment
    import actdyn.environment
    import actdyn.environment.action
    import actdyn.environment.observation
    import actdyn.environment.vectorfield
    import actdyn.metrics
    import actdyn.metrics.cost
    import actdyn.metrics.uncertainty
    import actdyn.models
    import actdyn.models.dynamics
    import actdyn.models.encoder
    import actdyn.policy
    import actdyn.policy.mpc
    import external.integrative_inference.src.modules as metadyn
    from actdyn.config import ExperimentConfig
    from actdyn.utils import save_load
    from actdyn.utils.experiment_helpers import setup_environment, setup_experiment
    from actdyn.utils.runtime import configure_runtime, ensure_dir
    from actdyn.utils.rollout import RecentRollout, Rollout, RolloutBuffer
    from actdyn.utils.helper import make_uniform_sampler
    from actdyn.utils.visualize import plot_vector_field, set_matplotlib_style
    from external.integrative_inference.experiments.model_utils import build_hypernetwork
    
    # Small constant to prevent numerical instability
    eps = 1e-6
    
    # Configure matplotlib for better aesthetics
    set_matplotlib_style()
    
    # Set random seed for reproducibility
    device = configure_runtime(seed=0)
    
    
    # % (Pretrain) Pretrain context dependent dynamics model
    z_sampler = make_uniform_sampler(-5.0, 5.0, 2)
    e_sampler = make_uniform_sampler([-3.0, -2.0], [-0.1, 2.0], 2)
    
    
    # % VAE Test with Experiment Config
    base_dir = os.path.join(os.path.dirname(__file__), "../../results", "CISS", "RBF")
    base_dir = ensure_dir(base_dir)
    dz, de, du, dy = 2, 2, 2, 50
    dt = 0.1
    alpha = 5
    action_strength = 0.5
    noise_scale = 0.01
    # torch.manual_seed(70)
    torch.manual_seed(7)
    e = e_sampler(1)
    a, b = e.reshape(-1)
    # ------------------------------------------------------------------------------
    # Action Model
    # ------------------------------------------------------------------------------
    action_model = actdyn.environment.action.IdentityActionEncoder(
        action_dim=du,
        latent_dim=dz,
        action_bounds=[-action_strength * alpha, action_strength * alpha],
        device=device,
    )
    # ------------------------------------------------------------------------------
    # Observation Model
    # ------------------------------------------------------------------------------
    obs_model = actdyn.environment.observation.LinearObservation(
        obs_dim=dy,
        latent_dim=dz,
        noise_scale=noise_scale,
        noise_type="gaussian",
        device=device,
    )
    # obs_model = actdyn.environment.observation.LogLinearObservation(
    #     obs_dim=dy,
    #     latent_dim=dz,
    #     noise_scale=0.1,
    #     noise_type="poisson",
    #     dt=dt,
    #     device=device,
    # )
    
    # C = obs_model.network[0].weight.detach()
    # C[:, 0] = torch.abs(C[:, 0])
    # # C[:, 0] = C[:, 0] * 3
    # # C[:, 1] = torch.abs(C[:, 1])
    # C[:, 1] = C[:, 1] * 2
    # # C = C / torch.norm(C, dim=1, keepdim=True)  # Normalize rows of C
    # C *= 1
    # mean_firing = 100
    # bias = torch.log(mean_firing * torch.ones(dy, device=device)) - 1 / 2 * torch.diag(C @ C.T)
    
    # obs_model.network[0].bias = nn.Parameter(bias)
    # obs_model.network[0].weight = nn.Parameter(C)
    
    
    # ------------------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------------------
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
    # ------------------------------------------------------------------------------
    # Decoder with Gaussian Noise
    # ------------------------------------------------------------------------------
    mapping = actdyn.models.decoder.LinearMapping(latent_dim=dz, obs_dim=dy, device=device)
    # mapping = actdyn.models.decoder.LogLinearMapping(latent_dim=dz, obs_dim=dy, dt=dt, device=device)
    noise = actdyn.models.decoder.GaussianNoise(obs_dim=dy, sigma=0.01, device=device)
    # noise = actdyn.models.decoder.PoissonNoise(obs_dim=dy, sigma=0.01, device=device)
    decoder = actdyn.models.Decoder(mapping=mapping, noise=noise, device=device)
    # ------------------------------------------------------------------------------
    # Model Components - Dynamics and model
    # ------------------------------------------------------------------------------
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
    # dynamics = actdyn.models.dynamics.MLPDynamics(
    #     state_dim=dz,
    #     hidden_dims=[64, 64],
    #     is_residual=True,
    #     device=device,
    # )
    # dynamics.logvar = nn.Parameter(torch.log(torch.ones(1, dz) * noise_scale).to(device))
    
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
    # model.set_params(e)
    
    # ------------------------------------------------------------------------------
    # Model Components - Policy
    # ------------------------------------------------------------------------------
    
    fisher_metric = actdyn.metrics.information.AOptimality(model=model)
    rnd_metric = actdyn.metrics.uncertainty.RandomNetworkDistillation(device=device)
    action_metric = actdyn.metrics.cost.ActionCost()
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
    step_policy = actdyn.policy.StepPolicy(action_space=env.action_space, step_size=50, device=device)
    random_policy = actdyn.policy.RandomPolicy(action_space=env.action_space, device=device)
    off_policy = actdyn.policy.OffPolicy(action_space=env.action_space, device=device)
    
    # ------------------------------------------------------------------------------
    # Model Components - Agent and Experiment
    # ------------------------------------------------------------------------------
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
    exp_config.training.total_steps = 20000
    experiment = actdyn.core.experiment.Experiment(
        agent=agent,
        config=exp_config,
    )
    
    decoder.set_params(obs_model)
    experiment.run()
    
    
    # print(f"True embedding: {e}, Learned embedding: {model.e['m']}")
    
    
    # # %
    # ro = save_load.load_and_concatenate_rollouts(os.path.join(base_dir, "rollouts"))
    # # ro = experiment.rollout
    # z = ro["env_state"]
    # z_hat = ro["model_state"]
    
    # plot_vector_field(dynamics, x_range=4)
    # plot_vector_field(duffing_env.dynamics, x_range=4)
    # plt.plot(to_np(z[0])[:, 0], to_np(z[0])[:, 1], alpha=0.7, label="true")
    # plt.plot(to_np(z_hat[0])[:, 0], to_np(z_hat[0])[:, 1], alpha=0.7, label="model")
    # plt.legend()
    # plt.show()
    # y = ro["obs"]
    # plt.plot(to_np(y[0])[:, :5])
    # plt.show()
    
    # # Create heatmap of Fisher Information
    # grid_size = 20
    # x = torch.linspace(-2, 2, grid_size)
    # y = torch.linspace(-2, 2, grid_size)
    # X, Y = torch.meshgrid(x, y)
    # Z = torch.zeros_like(X)
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         pos = torch.tensor([[X[i, j], Y[i, j]]], device=device)
    #         Z[i, j] = torch.log(J(pos)).item()
    # plt.figure(figsize=(6, 5))
    # plt.contourf(X.cpu(), Y.cpu(), Z.cpu(), levels=50, cmap="viridis")
    # plt.colorbar(label="Fisher Information")
    # plt.title("Fisher Information Heatmap")
    # plt.xlabel("x1")
    # plt.ylabel("x2")
    # plt.show()

if __name__ == "__main__":
    main()
