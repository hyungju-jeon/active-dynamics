#%%

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import torch.nn.functional as F
import numpy as np

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.torch_helper import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer


class CircularCartPole(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        base = env.unwrapped  # true CartPoleEnv

        # disable pole-angle termination by setting its threshold to infinity
        base.theta_threshold_radians = float('inf')

        # Cart limits from the underlying env
        self.x_threshold = base.x_threshold
        self.theta_threshold = base.theta_threshold_radians

        # track length and corresponding circle radius
        self.length = 2 * self.x_threshold
        self.radius = self.length / (2 * np.pi)

        # new obs: [cosθ, sinθ, θ̇, pole_angle, pole_ang_vel]
        low  = np.array([-1.0, -1.0, -np.inf, -self.theta_threshold, -np.inf], dtype=np.float32)
        high = np.array([ 1.0,  1.0,  np.inf,  self.theta_threshold,  np.inf], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # continuous action 0–1
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)

    def _wrap_state(self):
        base = self.env.unwrapped
        x, x_dot, theta, theta_dot = base.state
        # wrap-around smoothly
        if x > self.x_threshold:
            x -= self.length
        elif x < -self.x_threshold:
            x += self.length
        base.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

    def _make_obs(self):
        x, x_dot, theta, theta_dot = self.env.unwrapped.state
        angle   = x / self.radius
        ang_vel = x_dot / self.radius
        return np.array([np.cos(angle),
                         np.sin(angle),
                         ang_vel,
                         theta,
                         theta_dot], dtype=np.float32)

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        # raw_obs = [x, x_dot, θ, θ̇]
        self.env.unwrapped.state = raw_obs.astype(np.float32)
        return self._make_obs(), info

    def step(self, action):
        # discretize continuous input
        discrete_a = int(action[0] > 0.5)
        obs, reward, terminated, truncated, info = self.env.step(discrete_a)

        # override termination of hitting the x-threshold
        # only terminate if the pole itself goes beyond its angle threshold
        x, _, theta, _ = self.env.unwrapped.state
        if abs(x) > self.x_threshold and abs(theta) <= self.theta_threshold:
            terminated = False

        # wrap the state for smooth physics continuity
        self._wrap_state()

        return self._make_obs(), reward, terminated, truncated, info
    
    
if __name__ == "__main__":
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # create circular CartPole environment
    env = CircularCartPole(gym.make('CartPole-v1'))
    obs_dim = env.observation_space.shape[0]   # 5-D
    action_dim = env.action_space.shape[0]     # 1-D continuous

    # latent dimension
    latent_dim = obs_dim

    action_bounds = (0.0, 1.0)

    # define SeqVAE components
    encoder = MLPEncoder(
        input_dim=obs_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_dims=[64, 64]
    )
    decoder = Decoder(
        LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
        GaussianNoise(output_dim=obs_dim, sigma=0.5),
        device=device
    )
    dynamics = LinearDynamics(state_dim=latent_dim, device=device)
    action_encoder = LinearActionEncoder(
        action_dim=action_dim,
        latent_dim=latent_dim,
        action_bounds=action_bounds,
        device=device
    )

    model = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
        device=device
    )

    # collect random rollouts
    num_samples = 500
    num_steps = 100
    buffer = RolloutBuffer(num_samples)

    for _ in range(num_samples):
        obs_seq = torch.zeros(num_steps+1, obs_dim, device=device)
        actions = torch.zeros(num_steps, action_dim, device=device)

        obs_raw, _ = env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)

        for t in range(num_steps):
            a_cont = env.action_space.sample()
            actions[t] = torch.from_numpy(a_cont).to(device)

            obs_raw, _, done, _, _ = env.step(a_cont)
            obs_seq[t+1] = torch.from_numpy(obs_raw).float().to(device)
            if done:
                break

        rollout = Rollout()
        seq_len = obs_seq.size(0) - 1
        for t in range(seq_len):
            rollout.add(
                obs=obs_seq[t].unsqueeze(0),
                action=actions[t].unsqueeze(0),
                next_obs=obs_seq[t+1].unsqueeze(0)
            )
        buffer.add(rollout)

    # train
    model.action_dim = action_dim
    model.train(
        list(buffer.as_batch(batch_size=64, shuffle=True)),
        optimizer="AdamW",
        n_epochs=1000,
    )


# %%

    # visualize latent trajectory from first rollout
    raw = buffer.buffer[0]
    data = raw.as_dict()
    obs = data['obs'].squeeze(1).unsqueeze(0).to(device)  # [1, T, obs_dim]

    with torch.no_grad():
        enc_z, *_ = model.encoder(obs)

    plt.figure()
    plt.plot(to_np(enc_z[0, :, 0]), to_np(enc_z[0, :, 1]), '-', label='latent')
    plt.title('Encoded Latent Trajectory (CartPole)')
    plt.legend()
    plt.show()

    # reconstruction comparison
    recon = model.decoder(enc_z)
    plt.figure()
    plt.plot(to_np(recon[0]), '-', label='reconstructed obs')
    plt.plot(to_np(obs.squeeze(0)), '--', label='true obs')
    plt.title('Observation Reconstruction (CartPole)')
    plt.legend(loc='upper right')
    plt.show()

# %%