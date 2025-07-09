# %%
# Testing SeqVAE components on OpenAI Gymnasium Pendulum environment

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import torch.nn.functional as F

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

if __name__ == "__main__":
    # create Pendulum environment
    gym_env = gym.make('Pendulum-v1')
    obs_dim = gym_env.observation_space.shape[0]  # 3-D observation
    action_dim = gym_env.action_space.shape[0]    # 1-D continuous action

    latent_dim = obs_dim  # choose latent dimension same as observation dim

    # define SeqVAE model components
    encoder = MLPEncoder(input_dim=obs_dim, latent_dim=latent_dim,
                         device=device, hidden_dims=[32, 32])
    decoder = Decoder(
        LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
        GaussianNoise(output_dim=obs_dim, sigma=0.5),
        device=device,
    )
    dynamics = LinearDynamics(state_dim=latent_dim, device=device)
    action_encoder = LinearActionEncoder(
        input_dim=action_dim, latent_dim=latent_dim, device=device
    )

    model = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
        device=device,
    )

    # generate random rollouts from Pendulum
    num_samples = 500
    num_steps = 200
    rollout_buffer = RolloutBuffer(num_samples)

    for _ in range(num_samples):
        obs_seq = torch.zeros(num_steps+1, obs_dim, device=device)
        actions = torch.zeros(num_steps, action_dim, device=device)

        obs_raw, _ = gym_env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)

        for t in range(num_steps):
            # sample a continuous action
            a = gym_env.action_space.sample()
            a_tensor = torch.from_numpy(a).float().to(device)
            actions[t] = a_tensor

            obs_next, _, done, _, _ = gym_env.step(a)
            obs_seq[t+1] = torch.from_numpy(obs_next).float().to(device)
            if done:
                break

        # build rollout
        rollout = Rollout()
        seq_length = obs_seq.size(0) - 1
        for t in range(seq_length):
            rollout.add(
                obs=obs_seq[t].unsqueeze(0),
                action=actions[t].unsqueeze(0),
                next_obs=obs_seq[t+1].unsqueeze(0),
            )
        rollout_buffer.add(rollout)

    # train model end-to-end
    model.action_dim = latent_dim   # latent action dimension
    model.train(
        list(rollout_buffer.as_batch(batch_size=64, shuffle=True)),
        optimizer="AdamW",
        n_epochs=20000,
    )

# %%
    # visualize one stored rollout in latent space vs observations
    raw_rollout = rollout_buffer.buffer[0]
    rollout_dict = raw_rollout.as_dict()

    obs = rollout_dict['obs']        # [T,1,obs_dim]
    obs_traj = obs.squeeze(1).unsqueeze(0).to(device)  # [1, T, obs_dim]

    with torch.no_grad():
        enc_z, *_ = model.encoder(obs_traj)

    plt.figure()
    # plot first two latent dims
    plt.plot(to_np(enc_z[0, :, 0]), to_np(enc_z[0, :, 1]), '-', label='latent')
    plt.title('Encoded Latent Trajectory')
    plt.legend()
    plt.show()

    # reconstruction plot
    recon = model.decoder(enc_z)
    plt.figure()
    plt.plot(to_np(recon[0]), '-', label='reconstructed obs')
    plt.plot(to_np(obs_traj[0]), '--', label='true obs')
    plt.title('Observation Reconstruction')
    plt.legend()
    plt.show()

# %%
# Quick reconstruction test for encoder + decoder
import torch
import torch.nn.functional as F

batch_size = 64
dummy_obs = torch.randn(batch_size, obs_dim, device=device)

# encode and decode
z, *_ = encoder(dummy_obs.unsqueeze(1))  # [B,1,D]
z = z.squeeze(1)
recon = decoder(z)

# compute MSE
loss = F.mse_loss(recon, dummy_obs)
print(f"Reconstruction MSE on random test batch: {loss.item():.6f}")

# visualize a few examples
n_show = 3
for i in range(n_show):
    plt.figure()
    plt.plot(dummy_obs[:,i].cpu().numpy(), '--', label='original')
    plt.plot(recon[:,i].detach().cpu().numpy() * 1000, '-', label='recon')
    plt.title(f"Sample {i} Recon")
    plt.legend()
    plt.show()

# %%
