#%%

import torch
from matplotlib import pyplot as plt
import gymnasium as gym
import numpy as np
import math
from gym import spaces, logger
from gym.utils import seeding
import pygame
import random

from actdyn.models.encoder import MLPEncoder
from actdyn.models.decoder import Decoder, GaussianNoise, LinearMapping, IdentityMapping
from actdyn.models.dynamics import LinearDynamics, MLPDynamics
from actdyn.models.model import SeqVae
from actdyn.environment.action import LinearActionEncoder
from actdyn.utils.torch_helper import to_np
from actdyn.utils.rollout import Rollout, RolloutBuffer
from torch.utils.tensorboard import SummaryWriter



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
        self.tau = 0.01  # seconds between state updates
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
        force = self.force_mag * float(action[0])
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


# %%
# K-step prediction

# configuration
RENDER = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# hyperparameters
num_samples = 500
num_steps = 200
val_split = 0.1
batch_size = 64
n_epochs = 1500
learning_rate = 1e-2

# logger
writer = SummaryWriter(log_dir="runs/seqvae_cartpole")

def collect_rollouts(env, num_samples, num_steps, mode="random", plotFirstRollout=False):
    """
    collects rollouts from the environment
    """

    buffer = RolloutBuffer(num_samples)
    example_phi = []
    example_theta = []

    while len(buffer) < num_samples:
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        obs_seq = torch.zeros(num_steps + 1, obs_dim, device=device)
        actions = torch.zeros(num_steps, action_dim, device=device)

        obs_raw = env.reset()
        obs_seq[0] = torch.from_numpy(obs_raw).float().to(device)
        done = False
        t = 0
        for t in range(num_steps):
            if mode == "random":
                a = env.action_space.sample()
            elif mode == "passive":
                a = np.zeros(action_dim, dtype=np.float32)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # a = env.action_space.sample()
            actions[t] = torch.from_numpy(a).float().to(device)
            obs_next, _, done, _ = env.step(a)
            obs_seq[t + 1] = torch.from_numpy(obs_next).float().to(device)

            # save phi and theta for first rollout only
            if plotFirstRollout and len(buffer) == 0:
                example_phi.append(obs_next[0])   # phi
                example_theta.append(obs_next[2]) # theta

            if done:
                break
        
        if t + 1 < num_steps:
            continue  # Skip short episodes

        rollout = Rollout()
        for i in range(num_steps):
            rollout.add(
                obs=obs_seq[i].unsqueeze(0),
                action=actions[i].unsqueeze(0),
                next_obs=obs_seq[i + 1].unsqueeze(0),
            )
        buffer.add(rollout)

    # plot phi and theta for first rollout
    if plotFirstRollout and example_phi and example_theta:
        plt.figure(figsize=(10, 5))
        plt.plot(example_phi, label="phi (cart angular position)")
        plt.plot(example_theta, label="theta (pole angle)")
        plt.xlabel("Time step")
        plt.ylabel("Angle (radians)")
        plt.legend()
        plt.title("Phi and Theta over Time (First Rollout)")
        plt.show()

    return buffer

def k_step_prediction(model, rollout, k, writer, epoch):
    """
    performs a K-step prediction on a single rollout and logs the results
    """

    with torch.no_grad():
        # get data from rollout
        obs_seq = rollout._data['obs'].to(device)
        action_seq = rollout._data['action'].to(device)
        
        # take first observation as starting point
        obs_initial = obs_seq[0].unsqueeze(0)
        
        # use encoder to get initial latent state
        z_initial, *_ = model.encoder(obs_initial)
        
        # initialize lists to store predictions
        predicted_z_seq = [z_initial]
        predicted_obs_seq = [model.decoder(z_initial)[0]]

        # simulate dynamics for k steps
        current_z = z_initial
        for i in range(k):
            # encode action and predict next latent statei 
            if i < len(action_seq):
                current_action = action_seq[i].unsqueeze(0)
                encoded_action = model.action_encoder(current_action)
                current_z = model.dynamics(current_z, encoded_action)
            else:
                # if we run out of actions, assume no action (or a zero action)
                current_z = model.dynamics(current_z)
            
            # append to list
            predicted_z_seq.append(current_z)
            
            # decode predicted latent state to observation
            predicted_obs_seq.append(model.decoder(current_z)[0])
            
        # convert lists to tensors
        predicted_obs_seq = torch.cat(predicted_obs_seq, dim=0)
        predicted_z_seq = torch.cat(predicted_z_seq, dim=0)

        # plot and log predictions
        fig, axes = plt.subplots(1, obs_dim, figsize=(20, 5))
        for i in range(obs_dim):
            axes[i].plot(to_np(obs_seq[:k+1, i]), label='True')
            axes[i].plot(to_np(predicted_obs_seq[:k+1, i]), label=f'Predicted (k={k})', linestyle='--')
            axes[i].set_title(f'Observation Dim {i}')
            axes[i].legend()
        fig.suptitle(f'K-step Prediction for Epoch {epoch} (k={k})', fontsize=16)
        plt.tight_layout()
        plt.show()
        writer.add_figure(f"K-step Prediction/k={k}", fig, epoch)
        plt.close(fig)

        # # plot latent states
        # latent_dim = predicted_z_seq.shape[1]
        # fig_latent, axes_latent = plt.subplots(1, latent_dim, figsize=(20, 5))
        # if latent_dim == 1:
        #     axes_latent = [axes_latent]
        # for i in range(latent_dim):
        #     axes_latent[i].plot(to_np(predicted_z_seq[:, i]), label='Predicted latent ' + str(i), color='orange')
        #     axes_latent[i].set_title(f'Latent Dim {i}')
        #     axes_latent[i].legend()
        # fig_latent.suptitle(f'Latent Trajectory for Epoch {epoch} (k={k})', fontsize=16)
        # plt.tight_layout()
        # plt.show()
        # writer.add_figure(f"Latent Trajectory/k={k}", fig_latent, epoch)
        # plt.close(fig_latent)

        writer.flush()


if __name__ == "__main__":

    # create environment and collect data
    env = ContinuousCartPoleEnv()
    rollout_buffer = collect_rollouts(env, num_samples, num_steps, mode="passive", plotFirstRollout=True)

    # split into train and validation
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

    # model components
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    latent_dim = obs_dim

    action_bounds = (-1.0, 1.0) # Corrected action bounds

    encoder = MLPEncoder(
        input_dim=obs_dim,
        latent_dim=latent_dim,
        device=device,
        hidden_dims=[64, 64]
    )
    decoder = Decoder(
        # LinearMapping(latent_dim=latent_dim, output_dim=obs_dim),
        IdentityMapping(device=device),
        GaussianNoise(output_dim=obs_dim, sigma=0.5),
        device=device
    )
    # dynamics = LinearDynamics(state_dim=latent_dim, device=device)
    dynamics = MLPDynamics(state_dim=latent_dim, hidden_dim=64, device=device)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.action_dim = action_dim 

    # training loop with logging
    train_losses, val_losses = [], []
    for epoch in range(1, n_epochs + 1):
        # training
        epoch_train = []
        for batch in train_buffer.as_batch(batch_size=batch_size, shuffle=True):
            obs = batch['obs'].to(device)
            action = batch.get('action')
            action = action.to(device) if action is not None else None

            optimizer.zero_grad()
            loss = model.compute_elbo(obs, u=action, beta=10.0)
            loss.backward()
            optimizer.step()
            epoch_train.append(loss.item())
        avg_train = sum(epoch_train) / len(epoch_train)

        # validation
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

        # log
        writer.add_scalar("Loss/Train", avg_train, epoch)
        writer.add_scalar("Loss/Val", avg_val, epoch)
        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}/{n_epochs} - Train Loss: {avg_train:.3f}, Val Loss: {avg_val:.3f}")
        
        # K-step prediction at the end of some epochs
        if epoch % 50 == 0:
            # Choose a random rollout from the validation set
            sample_rollout = random.choice(val_rollouts)
            k_step_prediction(model, sample_rollout, k=50, writer=writer, epoch=epoch)
            
    writer.close()

    # plot losses
    starting_epoch = 0
    epochs = range(1, n_epochs + 1)
    plt.figure()
    plt.plot(epochs[starting_epoch:], train_losses[starting_epoch:], label="Train ELBO")
    plt.plot(epochs[starting_epoch:], val_losses[starting_epoch:], label="Val ELBO")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.title("Cartpole: Training and Validation ELBO over Epochs")
    plt.show()

