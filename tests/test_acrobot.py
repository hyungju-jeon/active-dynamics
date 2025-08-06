# %%
# Testing SeqVAE components on OpenAI Gymnasium Acrobot environment

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import torch.nn.functional as F
import numpy as np
import pygame

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, LinearMapping, GaussianNoise
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.torch_helper import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer

class AcrobotContinuousWrapper(gym.Wrapper):
    def __init__(self, env, max_torque=1.0, screen_width=600, screen_height=600):
        super().__init__(env)
        self.max_torque = max_torque
        # override action space with continuous Box
        self.action_space = gym.spaces.Box(
            low=-self.max_torque,
            high=self.max_torque,
            shape=(1,),
            dtype=np.float32
        )

        global RENDER
        if RENDER:
            # Pygame init
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption('Acrobot Continuous Control')
            self.clock = pygame.time.Clock()
            self.screen_width = screen_width
            self.screen_height = screen_height
            # scaling
            self.origin = np.array([screen_width // 2, screen_height // 4])
            self.scale = screen_height // 4  # scaling link lengths

    def step(self, action):
        torque = np.clip(action[0], -self.max_torque, self.max_torque)
        return self._step_continuous(torque)

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    def _step_continuous(self, torque):
        # extract the raw Acrobot env
        base = self.env.unwrapped

        # print(base.__dict__)

        # physics params
        dt = base.dt
        m1, m2 = base.LINK_MASS_1, base.LINK_MASS_2
        l1 = base.LINK_LENGTH_1
        lc1, lc2 = base.LINK_COM_POS_1, base.LINK_COM_POS_2
        I1 = base.LINK_MOI
        I2 = base.LINK_MOI
        g = 9.8 # i think this is the value they use originally? can't find inside "base"
        # current state
        a1, a2, a1_dot, a2_dot = base.state

        # equations of motion
        d1 = (m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*np.cos(a2)) + I1 + I2)
        d2 = m2*(lc2**2 + l1*lc2*np.cos(a2)) + I2
        phi2 = m2*lc2*g*np.cos(a1 + a2 - np.pi/2)
        phi1 = (-m2*l1*lc2*a2_dot**2*np.sin(a2)
                -2*m2*l1*lc2*a1_dot*a2_dot*np.sin(a2)
                + (m1*lc1 + m2*l1)*g*np.cos(a1 - np.pi/2)
                + phi2)
        # accelerations
        a2_ddot = (torque + d2/d1*phi1 - phi2) / (m2*lc2**2 + I2 - d2**2/d1)
        a1_ddot = -(d2*a2_ddot + phi1) / d1
        # integrate
        a1_dot += dt * a1_ddot
        a2_dot += dt * a2_ddot
        a1 += dt * a1_dot
        a2 += dt * a2_dot
        base.state = np.array([a1, a2, a1_dot, a2_dot])

        # get observations, reward, done
        ob = base._get_ob()
        terminal = False # never done in continuous setup
        reward = -1.0 if not terminal else 0.0
        # Gymnasium API: obs, reward, done, truncated, info
        return ob, reward, terminal, False, {}

    def render(self, mode='human'):
        global RENDER
        if not RENDER:
            return

        # clear screen
        self.screen.fill((255, 255, 255))
        # get angles
        a1, a2, *_ = self.env.unwrapped.state
        # link lengths scaled
        L1 = self.env.unwrapped.LINK_LENGTH_1 * self.scale
        L2 = self.env.unwrapped.LINK_LENGTH_2 * self.scale
        # compute positions
        p1 = self.origin + np.array([L1 * np.sin(a1), L1 * np.cos(a1)])
        p2 = p1 + np.array([L2 * np.sin(a1 + a2), L2 * np.cos(a1 + a2)])
        # draw links
        pygame.draw.line(self.screen, (0, 0, 0), self.origin, p1, 4)
        pygame.draw.circle(self.screen, (0, 0, 255), self.origin.astype(int), 6)
        pygame.draw.line(self.screen, (0, 0, 0), p1, p2, 4)
        pygame.draw.circle(self.screen, (255, 0, 0), p1.astype(int), 6)
        pygame.draw.circle(self.screen, (0, 255, 0), p2.astype(int), 6)
        # update
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        return self.env.close()


RENDER = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# Define environment parameters for the test
gym_env = AcrobotContinuousWrapper(gym.make('Acrobot-v1'))
obs_dim = gym_env.observation_space.shape[0]  # 6-D observation
action_dim = gym_env.action_space.shape[0]    # 1-D action (torque)
latent_dim = obs_dim
action_bounds = (-gym_env.max_torque, gym_env.max_torque)

# Define SeqVAE model components
encoder = MLPEncoder(input_dim=obs_dim, latent_dim=latent_dim, device=device, hidden_dims=[1])
decoder = Decoder(
    LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
    GaussianNoise(output_dim=obs_dim, sigma=0.5),
    device=device,
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
    device=device,
)

if __name__ == "__main__":
    # generate random rollouts from Acrobot
    num_samples = 500
    num_steps = 100
    rollout_buffer = RolloutBuffer(num_samples)

    for _ in range(num_samples):
        obs_seq = torch.zeros(num_steps+1, obs_dim, device=device)
        actions = torch.zeros(num_steps, action_dim, device=device)

        # reset environment and record initial observation
        obs_raw, _ = gym_env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)

        # collect trajectory
        for t in range(num_steps):
            a_cont = gym_env.action_space.sample()
            a_tensor = torch.from_numpy(a_cont).float().to(device)
            actions[t] = a_tensor

            obs_raw, reward, done, truncated, info = gym_env.step(a_cont)
            obs_seq[t+1] = torch.from_numpy(obs_raw).float().to(device)
            if done or truncated:
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
        n_epochs=1000,
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
