# %%
# Test each of the components of the model (encoder, dynamics, decoder) and combined SeqVAE

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import (
    Decoder,
    IdentityMapping,
    GaussianNoise,
)
from actdyn.models.dynamics import RBFDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import IdentityActionEncoder
from actdyn.environment.observation import (
    IdentityObservation,
)
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.core.agent import Agent
from actdyn.models import VAEWrapper
from actdyn.utils.torch_helper import to_np
from actdyn.utils.rollout import Rollout
from actdyn.utils.visualize import plot_vector_field
from actdyn.policy.policy import policy_from_str

# Set default device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

torch.manual_seed(1)

if __name__ == "__main__":
    # Define the environment
    vf = VectorFieldEnv(
        dynamics_type="limit_cycle", noise_scale=1e-3, device=device, dt=1
    )
    # obs_model = LogLinearObservation(latent_dim=2, obs_dim=50, device=device)
    obs_model = IdentityObservation(latent_dim=2, obs_dim=2, device=device)
    action_model = IdentityActionEncoder(
        action_dim=2, latent_dim=2, action_bounds=(-0.5, 0.5), device=device
    )
    env = GymObservationWrapper(
        env=vf, obs_model=obs_model, action_model=action_model, device=device
    )

    # Define the model
    encoder = MLPEncoder(input_dim=2, latent_dim=2, device=device)
    # decoder = Decoder(
    #     LogLinearMapping(latent_dim=2, output_dim=50),
    #     GaussianNoise(output_dim=50, sigma=1.0),
    #     device=device,
    # )
    decoder = Decoder(
        IdentityMapping(latent_dim=2, output_dim=2),
        GaussianNoise(output_dim=2, sigma=1.0),
        device=device,
    )
    rbf_grid_x = torch.linspace(-5, 5, 25)
    rbf_grid_y = torch.linspace(-5, 5, 25)
    rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="ij")  # [H, W]
    rbf_grid_pts = torch.stack([rbf_xx.flatten(), rbf_yy.flatten()], dim=1)
    dynamics = RBFDynamics(
        num_centers=625, state_dim=2, centers=rbf_grid_pts, device=device
    )
    action_encoder = IdentityActionEncoder(
        action_dim=2, latent_dim=2, action_bounds=(-0.5, 0.5), device=device
    )

    # Copy weights from obs_model to decoder mapping and freeze them
    # decoder.mapping.network[0].weight.data = obs_model.network[0].weight.data.clone()
    # decoder.mapping.network[0].bias.data = obs_model.network[0].bias.data.clone()
    # for param in decoder.mapping.network.parameters():
    #     param.requires_grad = False

    vae = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
        device=device,
    )
    model = VAEWrapper(vae, env.observation_space, env.action_space, device=device)
    policy_cls = policy_from_str("lazy")
    policy = policy_cls(env.action_space)

    # Pretrain
    # Generate data
    num_samples = 100
    num_steps = 100
    state_dim = 2
    x0 = torch.rand(num_samples, 1, state_dim, device=device) * 5 - 2.5
    z = vf.generate_trajectory(x0, num_steps)
    y = obs_model.observe(z)

    # Create rollout buffer
    # rollout_buffer = RolloutBuffer(num_samples)
    # for i in range(num_samples):
    #     rollout = Rollout()
    #     for t in range(num_steps):
    #         rollout.add(
    #             obs=y[i, t].unsqueeze(0),
    #             action=torch.randn(1, 2, device=device),
    #             env_state=z[i, t].unsqueeze(0),
    #         )
    #     rollout_buffer.add(rollout)

    # Train the model
    # model.train(
    #     list(rollout_buffer.as_batch(batch_size=64, shuffle=True)),
    #     optimizer="AdamW",
    #     n_epochs=50,
    #     lr=1e-4,
    # )
    plot_vector_field(model.model.dynamics)

    agent = Agent(env, model, policy, device=device)

    # %%
    # Start experiment
    agent.reset()
    print(agent._env_state)
    exp_rollout = Rollout(device=device)
    for i in tqdm(range(50000)):
        action = agent.plan()

        transition, done = agent.step(action)
        if i > 20:
            agent.train_model(
                optimizer="SGD",
                lr=1e-3,
                weight_decay=1e-5,
                n_epochs=1,
                verbose=False,
            )
        exp_rollout.add(**transition)
        with torch.no_grad():
            if i > 0 and i % 1000 == 0:
                z = torch.stack(exp_rollout["model_state"])
                z_true = torch.stack(exp_rollout["env_state"])
                plot_vector_field(agent.model_env.model.dynamics)
                plt.plot(to_np(z[-100:, 0]), to_np(z[-100:, 1]), label="Model State")
                plt.plot(
                    to_np(z_true[-100:, 0]),
                    to_np(z_true[-100:, 1]),
                    label="True State",
                )
                plt.legend()
                plt.show()

    exp_rollout.finalize()
