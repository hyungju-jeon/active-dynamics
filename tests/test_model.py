# %%
# Test each of the components of the model (encoder, dynamics, decoder) and combined SeqVAE

import torch
from actdyn.environment.env_wrapper import GymObservationWrapper
from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import *
from actdyn.models.dynamics import *
from actdyn.models.model import SeqVae
from actdyn.environment.action import *
from torch.utils.data import DataLoader
from actdyn.environment.observation import *
from actdyn.environment.vectorfield import *
from actdyn.utils.helpers import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer


def test_encoder():
    encoder = MLPEncoder(input_dim=10, latent_dim=2)
    z = encoder(x)
    assert z.shape == (10, 2)


if __name__ == "__main__":
    encoder = MLPEncoder(input_dim=5, latent_dim=2)
    decoder = make_linear_gaussian_decoder(latent_dim=2, output_dim=5)
    dynamics = LinearDynamics(state_dim=2)
    action_encoder = LinearActionEncoder(input_dim=3, latent_dim=2)
    model = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
    )
    vf = VectorFieldEnv(dynamics_type="limit_cycle")
    obs_model = LinearObservation(dz=2, dy=5)
    action_model = LinearActionEncoder(input_dim=3, latent_dim=2)
    env = GymObservationWrapper(env=vf, obs_model=obs_model, action_model=action_model)

    rollout_buffer = RolloutBuffer(1000)
    for i in range(1000):
        rollout = Rollout()
        obs, info = env.reset()
        obs = obs.unsqueeze(0)
        for i in range(500):
            action = torch.zeros(1, 3)
            next_obs, reward, terminate, done, info = env.step(action)
            transition = {
                "obs": obs,
                "action": action,
                "env_state": info["latent_state"],
                "next_obs": next_obs,
            }
            rollout.add(**transition)
            obs = next_obs
        rollout.finalize()
        rollout_buffer.add(rollout)

    model.train(rollout_buffer.flat, n_epochs=5000)
    # %%
    # Compare each component of the model with the true dynamics after training
    from matplotlib import pyplot as plt

    y = rollout_buffer.flat["obs"][:1, :, :]
    z = rollout_buffer.flat["env_state"][:1, :, :]
    plt.plot(
        to_np(model.encoder(y)[0][:, :, 0]).T,
        to_np(model.encoder(y)[0][:, :, 1]).T,
        "-",
    )
    plt.plot(to_np(z[:, :, 0].T), to_np(z[:, :, 1].T), "-")
    plt.show()

    recon = model.decoder(model.encoder(y)[0])
    plt.plot(to_np(recon[0, :, :2]), "-")
    plt.plot(to_np(y[0, :, :2]), "--")
    plt.show()
    # %%
