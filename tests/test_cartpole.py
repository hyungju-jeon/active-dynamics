#%%

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import math
from gym import spaces, logger
from gym.utils import seeding
import pygame

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping
from actdyn.models.dynamics import LinearDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.torch_helper import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer


class ContinuousCartPoleEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 30.0
        self.tau = 0.02  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # angle at which to fail the episode
        self.theta_threshold_radians = float('inf')  # no angle limit
        self.x_threshold = 2.4
        self.track_length = 2 * self.x_threshold
        # radius of equivalent circle: circumference = track_length
        self.radius = self.track_length / (2 * math.pi)

        # action and observation spaces
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )
        # phi unbounded, phi_dot, theta, theta_dot
        high = np.array([np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max,
                         np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # pygame renderer
        self._pygame_inited = False
        self.viewer = None

        # RNG
        self.seed()
        self.state = None

    def _init_pygame(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Circular CartPole')
        self.clock = pygame.time.Clock()

        # world-to-pixel scaling for linear track
        self.x_scale = self.screen_width / (self.track_length * 1.2)
        self.y_scale = self.screen_height / (self.length * 2 + 1)
        self.cart_w_pix = 0.4 * self.x_scale
        self.cart_h_pix = 0.2 * self.y_scale
        self.track_y = int(self.screen_height * 0.8)
        self._pygame_inited = True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def stepPhysics(self, force):
        phi, phi_dot, theta, theta_dot = self.state
        # compute linear acceleration xacc from CartPole eqs
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        # convert linear accel to angular accel on circle
        phiacc = xacc / self.radius

        # integrate
        phi = phi + self.tau * phi_dot
        phi_dot = phi_dot + self.tau * phiacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        return (phi, phi_dot, theta, theta_dot)

    def step(self, action):
        assert self.action_space.contains(action), \
            "%r (%s) invalid" % (action, type(action))
        force = self.force_mag * float(action)
        self.state = self.stepPhysics(force)

        # # unpack for debug print
        # phi, phi_dot, theta, theta_dot = self.state
        # print(f"[DEBUG] phi={phi:.3f}, phi_dot={phi_dot:.3f}, theta={theta:.3f}, theta_dot={theta_dot:.3f}")

        # never done in continuous setup
        reward = 0.0
        done = False
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        # start near phi=0, small velocities
        phi = 0.0
        phi_dot, theta, theta_dot = self.np_random.uniform(-0.05, 0.05, size=(3,))
        self.state = (phi, phi_dot, theta, theta_dot)
        return np.array(self.state, dtype=np.float32)

    def render(self, mode='human'):
        global RENDER
        if not RENDER:
            return
        if self.state is None:
            return

        phi, phi_dot, theta, theta_dot = self.state
        if not self._pygame_inited:
            self._init_pygame()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        self.screen.fill((255, 255, 255))
        # draw track line
        pygame.draw.line(
            self.screen, (0,0,0),
            (0, self.track_y),
            (self.screen_width, self.track_y), 2)
        # compute wrapped linear position s in [0, track_length)
        s = (phi % (2*math.pi)) * self.radius
        # center to world x
        x_centered = s - self.x_threshold
        cart_x = int(self.screen_width/2 + x_centered * self.x_scale)
        cart_y = self.track_y - int(self.cart_h_pix)
        # draw cart
        cart_rect = pygame.Rect(
            cart_x - self.cart_w_pix/2, cart_y,
            self.cart_w_pix, self.cart_h_pix)
        pygame.draw.rect(self.screen, (50,100,200), cart_rect)
        # draw pole
        pole_x0, pole_y0 = cart_x, cart_y
        pole_x1 = pole_x0 + self.length * math.sin(theta) * self.x_scale
        pole_y1 = pole_y0 - self.length * math.cos(theta) * self.y_scale
        pygame.draw.line(
            self.screen, (200,50,50),
            (pole_x0, pole_y0), (pole_x1, pole_y1), 6)
        pygame.display.flip()
        self.clock.tick(int(1.0/self.tau))

    def close(self):
        global RENDER
        if self._pygame_inited and RENDER:
            pygame.quit()


RENDER = False

if __name__ == "__main__":
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_device(device)

    # create circular CartPole environment
    env = ContinuousCartPoleEnv()
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

        obs_raw = env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)

        for t in range(num_steps):
            a_cont = env.action_space.sample()
            actions[t] = torch.from_numpy(a_cont).to(device)

            obs_raw, _, done, _, = env.step(a_cont)
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