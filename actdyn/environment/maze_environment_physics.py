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
                 max_engine_acc=25.0,
                 wall_thickness = 4.0,
                 lidar_num_beams=8,
                 lidar_range=50.0,
                 lidar_noise=0.0,
                 render_mode=None,
                 wind_field=None,
                 wind_scale=1.0):

        super().__init__()
        self.cell_size = cell_size
        self.wall_thickness = wall_thickness
        self.max_thrust = max_thrust
        self.max_engine_acc = max_engine_acc
        self.render_mode = render_mode
        self.agent_radius = int(0.4*self.cell_size)
        self.wind_field = wind_field
        self.wind_scale = wind_scale

        # LiDAR parameters
        self.lidar_num_beams = lidar_num_beams
        self.lidar_range = lidar_range
        self.lidar_noise = lidar_noise
        # beam angles evenly over [-FOV/2, FOV/2]
        self.lidar_fov = np.pi / 2
        self.lidar_angles = np.linspace(-self.lidar_fov/2,
                                         self.lidar_fov/2,
                                         self.lidar_num_beams)

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
        self.mass = 1.0
        self.inertia = 0.1
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
        self.state = np.array([*self.start_pos, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return self.state.copy(), {}
    
    def _get_obs(self):
        # compute LiDAR readings
        lidar = self._compute_lidar()  # shape (N,)
        # concatenate state and lidar
        return np.concatenate([self.state.copy(), lidar])

    def _compute_lidar(self):
        x, y, theta = self.state[0], self.state[1], self.state[2]
        readings = np.zeros(self.lidar_num_beams, dtype=np.float32)
        for i, rel_ang in enumerate(self.lidar_angles):
            beam_ang = theta + rel_ang
            dist = 0.0
            step = self.cell_size * 0.2  # step size
            while dist < self.lidar_range:
                rx = x + dist * np.sin(beam_ang)
                ry = y + dist * np.cos(beam_ang)
                if self._check_lidar_collision(rx, ry):
                    break
                dist += step
            # add noise
            if self.lidar_noise > 0:
                dist += self.np_random.normal(0, self.lidar_noise)
            readings[i] = min(dist, self.lidar_range)
        return readings
    
    def _check_collision(self, x, y):
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
    
    def _check_lidar_collision(self, x, y):
        i = int(np.floor(x / self.cell_size))
        j = int(np.floor(y / self.cell_size))
        if (i < 0 or i >= self.width or
            j < 0 or j >= self.height or
            self.maze[i, j] == 1):
            return True
        return False


    def step(self, action):
        target_l, target_r = np.clip(action, 0, self.max_thrust)
        x, y, theta, _, _, omega, current_l, current_r = self.state

        # update thrust with acceleration limits
        diff_l = target_l - current_l
        diff_r = target_r - current_r
        max_delta = self.max_engine_acc * self.dt
        new_l = current_l + np.clip(diff_l, -max_delta, max_delta)
        new_r = current_r + np.clip(diff_r, -max_delta, max_delta)

        # velocity forward
        vel = (new_r + new_l) / 2.0

        # angular rate
        omega = (new_r - new_l) / (2.0 * self.agent_radius)

        # wind effect
        if self.wind_field is not None:
            wx, wy = self.wind_field.get_wind(x, y)
        else:
            wx, wy = 0.0, 0.0
        wx *= self.wind_scale
        wy *= self.wind_scale

        # compute desired displacements
        dx = wx * self.dt + vel * np.sin(theta) * self.dt
        dy = wy * self.dt + vel * np.cos(theta) * self.dt

        # collision check with sliding along walls
        new_x = x + dx
        if self._check_collision(new_x, y):
            new_x = x  # block x movement
        new_y = y + dy
        if self._check_collision(new_x, new_y):
            new_y = y # block y movement

        # update angle
        new_theta = theta + omega * self.dt
        new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi

        # update velocities
        new_vx = vel * np.sin(new_theta)
        new_vy = vel * np.cos(new_theta)

        self.state = np.array([new_x, new_y, new_theta,
                               new_vx, new_vy, omega,
                               new_l, new_r], dtype=np.float32)

        # compute reward/done
        obs = self._get_obs()
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
        x,y,theta,*_ = self.state
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
            
        # LiDAR beams visualization
        for rel_ang in self.lidar_angles:
            beam_ang = theta + rel_ang
            dist = 0.0
            step = self.cell_size * 0.02
            while dist < self.lidar_range:
                rx = x + dist * np.sin(beam_ang)
                ry = y + dist * np.cos(beam_ang)
                if self._check_lidar_collision(rx, ry): break
                dist += step
            end = (int(x + dist * np.sin(beam_ang)), int(y + dist * np.cos(beam_ang)))
            pygame.draw.line(self.screen, (255,165,0), (int(x), int(y)), end, 1)

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
    # wind = WindField(dynamics_type="limit_cycle")
    # env.wind_field = wind
    # env.wind_scale = 0.00001  # scale the wind effect
    
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
# python -m actdyn.environment.maze_environment_physics