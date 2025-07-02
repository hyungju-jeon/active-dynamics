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
from actdyn.core.agent import Agent
from actdyn.policy import policy_factory
from actdyn.models import VAEWrapper

# Set default device to cuda if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


if __name__ == "__main__":
    # Define the environment
    vf = VectorFieldEnv(
        dynamics_type="limit_cycle", noise_scale=1e-2, device=device, dt=1
    )
    obs_model = LinearObservation(latent_dim=2, obs_dim=5, device=device)
    action_model = LinearActionEncoder(input_dim=2, latent_dim=2, device=device)
    env = GymObservationWrapper(env=vf, obs_model=obs_model, action_model=action_model)

    # Define the model
    encoder = MLPEncoder(input_dim=5, latent_dim=2, device=device)
    decoder = Decoder(
        LinearMapping(latent_dim=2, output_dim=5),
        GaussianNoise(output_dim=5, sigma=1.0),
        device=device,
    )
    dynamics = LinearDynamics(state_dim=2, device=device)
    action_encoder = LinearActionEncoder(input_dim=2, latent_dim=2, device=device)

    # Copy weights from obs_model to decoder mapping and freeze them
    decoder.mapping.network.weight.data = obs_model.network.weight.data.clone()
    decoder.mapping.network.bias.data = obs_model.network.bias.data.clone()
    for param in decoder.mapping.network.parameters():
        param.requires_grad = False

    vae = SeqVae(
        encoder=encoder,
        decoder=decoder,
        dynamics=dynamics,
        action_encoder=action_encoder,
        device=device,
    )
    model = VAEWrapper(vae, env.observation_space, env.action_space, device=device)
    policy_cls = policy_factory("lazy")
    policy = policy_cls(env.action_space)
    agent = Agent(env, model, policy, device=device)

    # Start experiment
    agent.reset()
    for i in range(50):
        action = agent.plan()
        agent.step(action)
        if i > 20:
            agent.train_model(optimizer="SGD")
