# %%

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import torch.nn.functional as F
import random

from actdyn.models.encoder import MLPEncoder, RNNEncoder
from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping, IdentityMapping
from actdyn.models.dynamics import LinearDynamics, MLPDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer
from torch.utils.tensorboard import SummaryWriter

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Hyperparameters
num_samples = 500
num_steps = 200
val_split = 0.1
batch_size = 64
n_epochs = 500
learning_rate = 1e-3

# TensorBoard logger
writer = SummaryWriter(log_dir="runs/seqvae_pendulum")

def collect_rollouts(env, num_samples, num_steps):
    buffer = RolloutBuffer(num_samples)
    while len(buffer) < num_samples:
        obs_seq = torch.zeros(num_steps + 1, env.observation_space.shape[0], device=device)
        actions = torch.zeros(num_steps, env.action_space.shape[0], device=device)

        obs_raw, _ = env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)
        done = False
        t = 0
        for t in range(num_steps):
            a = env.action_space.sample()
            actions[t] = torch.from_numpy(a).float().to(device)
            obs_next, _, done, _, _ = env.step(a)
            obs_seq[t + 1] = torch.from_numpy(obs_next).float().to(device)
            if done:
                break

        if t + 1 < num_steps:
            continue  # skip short episodes

        rollout = Rollout()
        for i in range(num_steps):
            rollout.add(
                obs=obs_seq[i].unsqueeze(0),
                action=actions[i].unsqueeze(0),
                next_obs=obs_seq[i + 1].unsqueeze(0),
            )
        buffer.add(rollout)
    return buffer

if __name__ == "__main__":
    # Create environment and collect data
    env = gym.make("Pendulum-v1")
    rollout_buffer = collect_rollouts(env, num_samples, num_steps)

    # Split into train and validation
    all_rollouts = list(rollout_buffer._buffer if hasattr(rollout_buffer, '_buffer') else rollout_buffer)
    random.shuffle(all_rollouts)
    n_val = int(len(all_rollouts) * val_split)
    val_rollouts = all_rollouts[:n_val]
    train_rollouts = all_rollouts[n_val:]


    train_buffer = RolloutBuffer(len(train_rollouts))
    for r in train_rollouts:
        train_buffer.add(r)

    val_buffer = RolloutBuffer(len(val_rollouts))
    for r in val_rollouts:
        val_buffer.add(r)

    # Model components
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = obs_dim

    encoder = RNNEncoder(
        input_dim=obs_dim, hidden_dim=32, latent_dim=latent_dim,
        rnn_type="gru", num_layers=1, device=device
    )
    decoder = Decoder(
        IdentityMapping(device=device),
        GaussianNoise(output_dim=obs_dim, sigma=1e-5),
        device=device,
    )
    dynamics = MLPDynamics(state_dim=latent_dim, device=device)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.action_dim = latent_dim

    # Training loop with logging
    train_losses, val_losses = [], []
    for epoch in range(1, n_epochs + 1):
        # Training
        model.encoder.train(); model.decoder.train(); model.dynamics.train(); model.action_encoder.train()
        epoch_train = []
        for batch in train_buffer.as_batch(batch_size=batch_size, shuffle=True):
            obs = batch['obs'].to(device)
            action = batch.get('action')
            action = action.to(device) if action is not None else None

            optimizer.zero_grad()
            loss = model.compute_elbo(obs, u=action)
            loss.backward()
            optimizer.step()
            epoch_train.append(loss.item())
        avg_train = sum(epoch_train) / len(epoch_train)

        # Validation
        model.encoder.eval(); model.decoder.eval(); model.dynamics.eval(); model.action_encoder.eval()
        epoch_val = []
        with torch.no_grad():
            for batch in val_buffer.as_batch(batch_size=batch_size, shuffle=False):
                obs = batch['obs'].to(device)
                action = batch.get('action')
                action = action.to(device) if action is not None else None
                loss = model.compute_elbo(obs, u=action)
                epoch_val.append(loss.item())
        avg_val = sum(epoch_val) / len(epoch_val)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        # Log
        writer.add_scalar("Loss/Train", avg_train, epoch)
        writer.add_scalar("Loss/Val",   avg_val,   epoch)
        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs} - Train Loss: {avg_train:.3f}, Val Loss: {avg_val:.3f}")

    writer.close()

    # Plot losses
    starting_epoch = 0
    epochs = range(1, n_epochs + 1)
    plt.figure()
    plt.plot(epochs[starting_epoch:], train_losses[starting_epoch:], label="Train ELBO")
    plt.plot(epochs[starting_epoch:], val_losses[starting_epoch:],   label="Val ELBO")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.title("Training and Validation ELBO over Epochs")
    plt.show()

# %%