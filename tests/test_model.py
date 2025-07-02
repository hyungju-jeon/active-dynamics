# %%
# Test each of the components of the model (encoder, dynamics, decoder) and combined SeqVAE

import torch
from matplotlib import pyplot as plt

from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, LinearMapping, GaussianNoise
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.environment.observation import LinearObservation
from actdyn.environment.vectorfield import VectorFieldEnv
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer

# Set default device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


if __name__ == "__main__":
    # Define the environment
    vf = VectorFieldEnv(
        dynamics_type="limit_cycle", noise_scale=1e-2, device=device, dt=1
    )
    obs_model = LinearObservation(latent_dim=2, obs_dim=5, device=device)
    action_model = LinearActionEncoder(input_dim=3, latent_dim=2, device=device)
    env = GymObservationWrapper(env=vf, obs_model=obs_model, action_model=action_model)

    # Define the model
    encoder = MLPEncoder(input_dim=5, latent_dim=2, device=device)
    decoder = Decoder(
        LinearMapping(latent_dim=2, output_dim=5),
        GaussianNoise(output_dim=5, sigma=1.0),
        device=device,
    )
    dynamics = LinearDynamics(state_dim=2, device=device)
    action_encoder = LinearActionEncoder(input_dim=3, latent_dim=2, device=device)

    # Copy weights from obs_model to decoder mapping and freeze them
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

    # Generate data
    num_samples = 500
    num_steps = 100
    state_dim = 2
    x0 = torch.rand(num_samples, 1, state_dim, device=device) * 5 - 2.5
    z = vf.generate_trajectory(x0, num_steps)
    y = obs_model.observe(z)

    # Create rollout buffer
    rollout_buffer = RolloutBuffer(num_samples)
    for i in range(num_samples):
        rollout = Rollout()
        for t in range(num_steps):
            rollout.add(
                obs=y[i, t].unsqueeze(0),
                action=torch.randn(1, 3, device=device),
                env_state=z[i, t].unsqueeze(0),
                next_obs=y[i, t + 1].unsqueeze(0),
            )
        rollout_buffer.add(rollout)

    # Train the model
    model.action_dim = 0
    model.train(
        list(rollout_buffer.as_batch(batch_size=16, shuffle=True)),
        optimizer="AdamW",
        n_epochs=1000,
    )

    # Compare each component of the model with the true dynamics after training
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
    plt.plot(to_np(recon[0, :, :2]), "-", label="reconstructed")
    plt.plot(to_np(y[0, :, :2]), "--", label="true")
    plt.legend()
    plt.show()
