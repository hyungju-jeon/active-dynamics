# %%

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import torch.nn.functional as F

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.torch_helper import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

if __name__ == "__main__":
    # create CartPole environment
    env = gym.make('CartPole-v1')
    obs_dim = env.observation_space.shape[0]  # 4-D observation
    action_space = env.action_space.n         # 2 discrete actions

    # latent dimension
    latent_dim = obs_dim

    action_bounds = (0, 1)

    # define SeqVAE components
    encoder = MLPEncoder(
        input_dim=obs_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_dims=[64, 64]  # you can adjust hidden sizes
    )
    decoder = Decoder(
        LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
        GaussianNoise(output_dim=obs_dim, sigma=0.5),
        device=device
    )
    dynamics = LinearDynamics(state_dim=latent_dim, device=device)
    action_encoder = LinearActionEncoder(
        action_dim=action_space,
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
        actions = torch.zeros(num_steps, action_space, device=device)

        # reset and record
        obs_raw, _ = env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)

        for t in range(num_steps):
            a_int = env.action_space.sample()
            a_onehot = F.one_hot(torch.tensor(a_int), num_classes=action_space).float().to(device)
            actions[t] = a_onehot

            obs_raw, _, done, _, _ = env.step(a_int)
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
    model.action_dim = 2  # latent action dim
    model.train(
        list(buffer.as_batch(batch_size=64, shuffle=True)),
        optimizer="AdamW",
        n_epochs=5000,  # adjust epochs as needed
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
    plt.legend()
    plt.show()

# %%