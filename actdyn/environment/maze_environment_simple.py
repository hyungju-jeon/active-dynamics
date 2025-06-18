import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from  actdyn.environment.windfield import WindField

class ContinuousMazeEnv(gym.Env):
    """
    Continuous 2D maze with a differential-drive boat (two side engines).
    Action = [thrust_left, thrust_right] ∈ [0,50]^2.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 maze_file,
                 cell_size=32,
                 max_thrust=50.0,
                 wall_thickness = 4.0,
                 render_mode=None,
                 wind_field=None,
                 wind_scale=1.0):

        super().__init__()
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.max_thrust = max_thrust
        self.render_mode = render_mode
        self.agent_radius = int(0.4*self.cell_size)
        self.wind_field = wind_field
        self.wind_scale = wind_scale

        # load maze
        with open(maze_file, 'r', encoding='utf-8') as f:
            lines = [line.rstrip('\n') for line in f]
        self.height = len(lines)
        self.width = len(lines[0])
        self.maze = np.array([[0 if ch==' ' else 1
                               for ch in line] for line in lines], dtype=np.int8).T

        # start/goal in cell coords
        start = (1,1)
        goal = (self.width-2, self.height-2)
        self.start_cell = np.array(start)
        self.goal_cell  = np.array(goal)

        # continuous start/goal in world coords
        self.start_pos = (self.start_cell + 0.5) * cell_size
        self.goal_pos  = (self.goal_cell  + 0.5) * cell_size

        # physics parameters
        self.dt = 1.0 / self.metadata["render_fps"]

        # state: [x, y, theta, vx, vy, omega]
        high = np.array([self.width*cell_size,
                         self.height*cell_size,
                         np.pi,
                         np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # action: left‐thrust, right‐thrust
        self.action_space = spaces.Box(0.0, max_thrust, shape=(2,), dtype=np.float32)

        # rendering
        self.screen = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # reset state to start, zero velocities
        self.state = np.array([*self.start_pos, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), {}
    
    def check_collision(self, x, y):
        min_x = x - self.agent_radius
        max_x = x + self.agent_radius
        min_y = y - self.agent_radius
        max_y = y + self.agent_radius

        # grid indices
        i_min = int(np.floor(min_x / self.cell_size))
        i_max = int(np.floor(max_x / self.cell_size))
        j_min = int(np.floor(min_y / self.cell_size))
        j_max = int(np.floor(max_y / self.cell_size))

        collision = False
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # if outside the maze bounds or hitting a wall cell
                if (i < 0 or i >= self.width or
                    j < 0 or j >= self.height or
                    self.maze[i, j] == 1):
                    collision = True
                    break
            if collision:
                break

        return collision


    def step(self, action):
        thrust_l, thrust_r = np.clip(action, 0, self.max_thrust)
        x, y, theta, *_ = self.state

        # velocity forward
        vel = (thrust_r + thrust_l) / 2.0

        # angular rate
        omega = (thrust_r - thrust_l) / (2.0 * self.agent_radius)

        # wind effect
        if self.wind_field is not None:
            wx, wy = self.wind_field.get_wind(x, y)
        else:
            wx, wy = 0.0, 0.0
        wx *= self.wind_scale
        wy *= self.wind_scale

        # integrate pose
        x_new = x + (wx if self.wind_field is not None else 0.0) + vel * np.sin(theta) * self.dt
        y_new = y + (wy if self.wind_field is not None else 0.0) + vel * np.cos(theta) * self.dt

        theta_new = theta + omega * self.dt

        # wrap angle to [-pi, pi]
        theta_new = (theta_new + np.pi) % (2*np.pi) - np.pi

        if self.check_collision(x_new, y_new):
            # reset to old pose, zero velocities
            x_new, y_new, omega = x, y, 0.0
            vel = 0.0

        # update state
        vx_new = vel * np.sin(theta_new)
        vy_new = vel * np.cos(theta_new)

        self.state = np.array([x_new, y_new, theta_new,
                               vx_new, vy_new, omega], dtype=np.float32)

        # compute reward/done
        dist = np.linalg.norm(self.state[:2] - self.goal_pos)
        done = dist < (0.5 * self.cell_size)
        reward = -dist
        obs = self.state.copy()
        info = {"distance": dist}
        return obs, reward, done, False, info


    def render(self):
        if self.render_mode is None:
            return

        # init
        if self.screen is None:
            size = (self.width*self.cell_size, self.height*self.cell_size)
            if self.render_mode=='human':
                self.screen = pygame.display.set_mode(size)
            else:
                self.screen = pygame.Surface(size)
            self.clock = pygame.time.Clock()

        # draw maze
        self.screen.fill((255,255,255))
        for cx in range(self.width):
            for cy in range(self.height):
                if self.maze[cx,cy]:
                    rect = pygame.Rect(cx*self.cell_size, cy*self.cell_size,
                                       self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen,(0,0,0),rect)

        # goal
        pygame.draw.circle(self.screen,(0,255,0),
                           self.goal_pos.astype(int), self.agent_radius)

        # boat
        x,y,theta,_,_,_ = self.state
        tip = (x + np.sin(theta) * self.agent_radius * 1.5,
                y + np.cos(theta) * self.agent_radius * 1.5)
        pygame.draw.circle(self.screen,(255,0,0),(int(x),int(y)), self.agent_radius)
        pygame.draw.line(self.screen,(0,0,255),(int(x),int(y)),(int(tip[0]),int(tip[1])),2)


        # wind vector
        if self.wind_field is not None:
            wx, wy = self.wind_field.get_wind(x, y)
            wx *= self.wind_scale
            wy *= self.wind_scale
            wind_tip = (x + wx * 10, y + wy * 10)
            pygame.draw.line(self.screen, (255, 165, 0), (int(x), int(y)),
                             (int(wind_tip[0]), int(wind_tip[1])), 2)

        # flip or return
        if self.render_mode=='human':
            pygame.display.flip()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()


if __name__ == '__main__':
    pygame.init()

    env = ContinuousMazeEnv('others/generated_mazes/maze_39x19_seed1_20250618_142109.txt',
                             render_mode='human')
    wind = WindField(dynamics_type="limit_cycle")
    env.wind_field = wind
    env.wind_scale = 0.00001  # scale the wind effect
    
    obs, _ = env.reset()
    done = False
    while not done:
        # handle events
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()

        # full thrust when pressed, zero otherwise
        left_thrust  = env.max_thrust if keys[pygame.K_a] else 0.0
        right_thrust = env.max_thrust if keys[pygame.K_d] else 0.0

        obs, reward, done, truncated, info = env.step([left_thrust, right_thrust])
        env.render()

    env.close()
    print('Finished!', info)

# RUN COMMAND:
# python -m actdyn.environment.maze_environment_simple