# %%
# Test each of the components of the model (encoder, dynamics, decoder) and combined SeqVAE

import torch
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import *
from actdyn.models.dynamics import *
from actdyn.models.model import SeqVae
from actdyn.environment.action import *
from actdyn.environment.observation import *
from actdyn.environment.vectorfield import *
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer
from matplotlib import pyplot as plt

# set default device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch defult to device
torch.set_default_device(device)


if __name__ == "__main__":
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

    decoder.mapping.network.weight.data = obs_model.network.weight.data.clone()
    decoder.mapping.network.bias.data = obs_model.network.bias.data.clone()
    for param in decoder.mapping.network.parameters():
        param.requires_grad = False
    model = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
        device=device,
    )

    # %%
    num_samples = 500
    num_steps = 100
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
        n_epochs=1000,
    )
    # %%
    # Compare each component of the model with the true dynamics after training
    y = rollout_buffer.flat["obs"]
    z = rollout_buffer.flat["env_state"]
    plt.plot(
        to_np(model.encoder(y)[0][:1, :, 0]).T,
        to_np(model.encoder(y)[0][:1, :, 1]).T,
        "-",
        label="reconstructed",
    )
    plt.plot(to_np(z[:1, :, 0].T), to_np(z[:1, :, 1].T), "-", label="true (mean)")
    plt.legend()
    plt.show()

    recon = model.decoder(model.encoder(y)[0])
    plt.plot(to_np(recon[0, :, :5]), "-", label="reconstructed")
    plt.plot(to_np(y[0, :, :5]), "--", label="true")
    plt.legend()
    plt.show()
    # %%
