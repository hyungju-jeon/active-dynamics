# %%
# Testing SeqVAE components on OpenAI Gymnasium Acrobot environment

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import torch.nn.functional as F

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, LinearMapping, GaussianNoise
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Define environment parameters for the test
gym_env = gym.make('Acrobot-v1')
obs_dim = gym_env.observation_space.shape[0]  # 6-D observation
action_space = gym_env.action_space.n         # 3 discrete actions
latent_dim = obs_dim

# Define SeqVAE model components
encoder = MLPEncoder(input_dim=obs_dim, latent_dim=latent_dim, device=device, hidden_dims=[1])
decoder = Decoder(
    LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
    GaussianNoise(output_dim=obs_dim, sigma=0.5),
    device=device,
)
dynamics = LinearDynamics(state_dim=latent_dim, device=device)
action_encoder = LinearActionEncoder(input_dim=action_space, latent_dim=latent_dim, device=device)

model = SeqVae(
    encoder=encoder,
    decoder=decoder,
    dynamics=dynamics,
    action_encoder=action_encoder,
    device=device,
)

if __name__ == "__main__":
    # generate random rollouts from Acrobot
    num_samples = 500
    num_steps = 100
    rollout_buffer = RolloutBuffer(num_samples)

    for _ in range(num_samples):
        obs_seq = torch.zeros(num_steps+1, obs_dim, device=device)
        actions = torch.zeros(num_steps, action_space, device=device)

        # reset environment and record initial observation
        obs_raw, _ = gym_env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)

        # collect trajectory
        for t in range(num_steps):
            a_int = gym_env.action_space.sample()
            a_onehot = F.one_hot(torch.tensor(a_int), num_classes=action_space).float().to(device)
            actions[t] = a_onehot

            obs_raw, _, done, _, _ = gym_env.step(a_int)
            obs_seq[t+1] = torch.from_numpy(obs_raw).float().to(device)
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
    model.action_dim = 2  # latent action dimension
    model.train(
        list(rollout_buffer.as_batch(batch_size=64, shuffle=True)),
        optimizer="AdamW",
        n_epochs=50000,
    )

# %%

    # visualize one stored rollout in latent space vs observations
    raw_rollout = rollout_buffer.buffer[0]  # first rollout
    rollout_dict = raw_rollout.as_dict()

    obs = rollout_dict['obs']        # tensor[T,1,obs_dim]
    obs_traj = obs.squeeze(1)        # now [T, obs_dim]
    obs_traj = obs_traj.unsqueeze(0) # now [1, T, obs_dim]
    obs_traj = obs_traj.to(device)   # move to GPU

    with torch.no_grad():
        enc_z, *_ = model.encoder(obs_traj)

    plt.figure()
    plt.plot(to_np(enc_z[0,:,0]), to_np(enc_z[0,:,1]), '-', label='latent')
    plt.title('Encoded Latent Trajectory')
    plt.legend()
    plt.show()

    # reconstruction plot for the same rollout
    recon = model.decoder(enc_z)
    plt.figure()
    plt.plot(to_np(recon[0]), '-', label='reconstructed obs')
    plt.plot(to_np(obs_traj[0]), '--', label='true obs')
    plt.title('Observation Reconstruction')
    plt.legend()
    plt.show()

# %%

# Quick reconstruction test for encoder + decoder

# generate a batch of random “observations”
batch_size = 64
dummy_obs = torch.randn(batch_size, obs_dim, device=device)

# encode to latent
z, *_ = encoder(dummy_obs.unsqueeze(1))  # [B,1,D] → [B,latent_dim]
z = z.squeeze(1)

# decode back to observation space
recon = decoder(z)

# compute simple MSE reconstruction loss
loss = F.mse_loss(recon, dummy_obs)
print(f"Reconstruction MSE on random test batch: {loss.item():.6f}")

# (optional) visualize a few examples

n_show = 5
for i in range(n_show):
    plt.figure()
    plt.plot(dummy_obs[:,i].cpu().numpy(), '--', label='original')
    plt.plot(recon[:,i].detach().cpu().numpy() * 1000, '-', label='recon')
    plt.title(f"Sample {i} Recon")
    plt.legend()
    plt.show()

# %%
