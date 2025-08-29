import torch
import gymnasium as gym
import numpy as np
import math
from gymnasium import spaces, logger
from typing import Optional
from gym.utils import seeding
import pygame
import random


class ContinuousCartPoleEnv(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self, dt=0.01, **kwargs):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 1  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 30.0
        self.tau = dt  # seconds between state updates
        self.min_action = -1.0
        self.max_action = 1.0

        # angle at which to fail the episode
        self.theta_threshold_radians = float("inf")  # no angle limit
        self.x_threshold = 2.4
        self.track_length = 2 * self.x_threshold
        # radius of equivalent circle: circumference = track_length
        self.radius = self.track_length / (2 * math.pi)

        # action and observation spaces
        self.min_action = -1.0
        self.max_action = 1.0
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        # observation structure:
        # [ sin(phi), cos(phi), phi_dot, sin(theta), cos(theta), theta_dot ]
        high = np.array(
            [
                1.0,  # sin(phi)
                1.0,  # cos(phi)
                np.finfo(np.float32).max,  # phi_dot
                1.0,  # sin(theta)
                1.0,  # cos(theta)
                np.finfo(np.float32).max,  # theta_dot
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # pygame renderer
        self._pygame_inited = False
        self.viewer = None

        self.seed()
        self.state = None

    def _init_pygame(self):
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Circular CartPole")
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

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        # thetaacc = (self.gravity * sintheta - costheta * temp) / (
        #     self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
        # )
        # xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        mass_sin = self.masscart + self.masspole * sintheta**2
        temp = force + self.polemass_length * theta_dot**2 * sintheta

        xacc = (temp + self.masspole * self.gravity * sintheta * costheta) / mass_sin
        thetaacc = -(self.total_mass * self.gravity * sintheta + costheta * temp) / (
            self.length * mass_sin
        )

        phiacc = xacc / self.radius

        phi_dot = phi_dot + self.tau * phiacc
        phi = phi + self.tau * phi_dot + self.tau
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        return (phi, phi_dot, theta, theta_dot)

    def _state_to_observation(self, state):
        """
        Convert internal true state (phi, phi_dot, theta, theta_dot)
        to observation: [sin(phi), cos(phi), phi_dot, sin(theta), cos(theta), theta_dot]
        """
        phi, phi_dot, theta, theta_dot = state
        # return np.array(
        #     [math.sin(phi), math.cos(phi), phi_dot, math.sin(theta), math.cos(theta), theta_dot],
        #     dtype=np.float32,
        # )
        return np.array(
            [math.sin(phi), math.cos(phi), phi_dot, math.sin(theta), math.cos(theta), theta_dot],
            dtype=np.float32,
        )

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        force = self.force_mag * float(action[0])
        self.state = self.stepPhysics(force)

        reward = 0.0
        done = False
        obs = self._state_to_observation(self.state)
        return obs, reward, done, False, {}

    def reset(self, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        # start near phi=0, small velocities
        phi = 0.0
        phi_dot, theta, theta_dot = self.np_random.uniform(-0.05, 0.05, size=(3,))
        self.state = (phi, phi_dot, theta, theta_dot)
        return self._state_to_observation(self.state), {}

    def render(self, mode="human"):
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
            self.screen, (0, 0, 0), (0, self.track_y), (self.screen_width, self.track_y), 2
        )
        # compute wrapped linear position s in [0, track_length)
        s = (phi % (2 * math.pi)) * self.radius
        # center to world x
        x_centered = s - self.x_threshold
        cart_x = int(self.screen_width / 2 + x_centered * self.x_scale)
        cart_y = self.track_y - int(self.cart_h_pix)
        # draw cart
        cart_rect = pygame.Rect(
            cart_x - self.cart_w_pix / 2, cart_y, self.cart_w_pix, self.cart_h_pix
        )
        pygame.draw.rect(self.screen, (50, 100, 200), cart_rect)
        # draw pole
        pole_x0, pole_y0 = cart_x, cart_y
        pole_x1 = pole_x0 + self.length * math.sin(theta) * self.x_scale
        pole_y1 = pole_y0 - self.length * math.cos(theta) * self.y_scale
        pygame.draw.line(self.screen, (200, 50, 50), (pole_x0, pole_y0), (pole_x1, pole_y1), 6)
        pygame.display.flip()
        self.clock.tick(int(1.0 / self.tau))

    def close(self):
        global RENDER
        if self._pygame_inited and RENDER:
            pygame.quit()


if __name__ == "__main__":
    env = ContinuousCartPoleEnv()
    RENDER = True
    for ep in range(100):
        obs = env.reset()
        for t in range(200):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
    env.close()
