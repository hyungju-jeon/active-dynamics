# %%
# Test each Policy

import torch
import gym
import numpy as np
from gym.spaces import Box
from matplotlib import pyplot as plt
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, LinearMapping, GaussianNoise
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import IdentityActionEncoder
from actdyn.environment.observation import LinearObservation
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.policy.random import RandomPolicy
from actdyn.policy.lazy import LazyPolicy
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer

# Set default device to cuda if available
use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"
torch.set_default_device(device)

if __name__ == "__main__":
    # --- Environment and Model Setup ---
    vf = VectorFieldEnv(
        dynamics_type="limit_cycle", noise_scale=1e-2, device=device, dt=1
    )
    obs_model = LinearObservation(latent_dim=2, obs_dim=5, device=device)
    action_model = IdentityActionEncoder(input_dim=2, latent_dim=2, device=device)
    env = GymObservationWrapper(
        env=vf, obs_model=obs_model, action_model=action_model, device=device
    )

    encoder = MLPEncoder(input_dim=5, latent_dim=2, device=device)
    decoder = Decoder(
        LinearMapping(latent_dim=2, output_dim=5),
        GaussianNoise(output_dim=5, sigma=1.0),
        device=device,
    )
    dynamics = LinearDynamics(state_dim=2, device=device)
    action_encoder = IdentityActionEncoder(input_dim=2, latent_dim=2, device=device)

    # Skipping decoder.mapping weight/bias copy and parameter freezing for simplicity

    model = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
        device=device,
    )

    # --- Rollout and Train Model ---
    num_samples = 100
    num_steps = 50
    rollout_buffer = RolloutBuffer(num_samples)
    for i in range(num_samples):
        rollout = Rollout(device=device)
        obs, info = env.reset()
        obs = obs.unsqueeze(0)
        for _ in range(num_steps):
            action = torch.zeros(1, 2)
            next_obs, reward, terminate, done, info = env.step(action)
            transition = {
                "obs": obs,
                "action": action,
                "env_state": info["latent_state"],
                "next_obs": next_obs,
            }
            rollout.add(**transition)
            obs = next_obs
        rollout_buffer.add(rollout)

    # Train the model
    model.train(
        list(rollout_buffer.to(device).as_batch(batch_size=16, shuffle=True)),
        optimizer="AdamW",
        n_epochs=100,
    )

    # --- Plot Model Reconstruction ---
    y = rollout_buffer.flat["obs"]
    z = rollout_buffer.flat["env_state"]
    plt.figure()
    plt.plot(
        to_np(model.encoder(y)[0][:1, :, 0]).T,
        to_np(model.encoder(y)[0][:1, :, 1]).T,
        "-",
        label="reconstructed",
    )
    plt.plot(to_np(z[:1, :, 0].T), to_np(z[:1, :, 1].T), "-", label="true (mean)")
    plt.legend()
    plt.title("Latent Trajectory: Model vs True")
    plt.show()

    recon = model.decoder(model.encoder(y)[0])
    plt.figure()
    plt.plot(to_np(recon[0, :, :5]), "-", label="reconstructed")
    plt.plot(to_np(y[0, :, :5]), "--", label="true")
    plt.legend()
    plt.title("Observation Reconstruction")
    plt.show()

    # --- Test and Plot RandomPolicy ---
    print("\nTesting RandomPolicy...")
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=float)
    random_policy = RandomPolicy(action_space)
    obs, info = env.reset()
    obs = obs.unsqueeze(0)
    actions = []
    for _ in range(num_steps):
        action = torch.tensor(random_policy.get_action(obs.squeeze(0)))
        actions.append(to_np(action))
        next_obs, reward, terminate, done, info = env.step(action.unsqueeze(0))
        obs = next_obs
    actions = np.stack(actions)
    plt.figure()
    plt.plot(actions)
    plt.title("RandomPolicy Actions")
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.show()

    # --- Test and Plot LazyPolicy ---
    print("\nTesting LazyPolicy...")
    lazy_policy = LazyPolicy(action_space)
    obs, info = env.reset()
    obs = obs.unsqueeze(0)
    actions = []
    for _ in range(num_steps):
        action = torch.tensor(lazy_policy.get_action(obs.squeeze(0)))
        actions.append(to_np(action))
        next_obs, reward, terminate, done, info = env.step(action.unsqueeze(0))
        obs = next_obs
    actions = np.stack(actions)
    plt.figure()
    plt.plot(actions)
    plt.title("LazyPolicy Actions")
    plt.xlabel("Step")
    plt.ylabel("Action Value")
    plt.show()
