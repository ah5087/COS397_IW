import numpy as np
import gym
from gym import spaces

class CellMigrationEnv(gym.Env):
    def __init__(self, initial_cell_density, illumination_pattern, t_max=100):
        super(CellMigrationEnv, self).__init__()

        # parameters
        self.t = 0
        self.t_max = t_max
        self.initial_cell_density = initial_cell_density
        self.illumination_pattern = illumination_pattern
        
        # action space: predict how cells move (-1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(101, 101), dtype=np.float32)
        
        # observation space: current cell density and illumination pattern
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 101, 101), dtype=np.float32)

        # initialize the current state
        self.state = {
            'cell_density': np.copy(initial_cell_density),
            'illumination_pattern': illumination_pattern
        }

    def step(self, action):
        self.t += 1

        # update density based on the action (predicted cell movements)
        new_cell_density = self.state['cell_density'] + action
        new_cell_density = np.clip(new_cell_density, 0, 1)  # clip btwn 0 and 1
        
        # update state
        self.state['cell_density'] = new_cell_density

        # reward based on how close the prediction is to the true cell movement
        reward = -np.mean((new_cell_density - self.true_cell_density())**2)

        done = self.t >= self.t_max
        return self._get_obs(), reward, done, {}

    def reset(self):
        self.t = 0
        self.state['cell_density'] = np.copy(self.initial_cell_density)
        return self._get_obs()

    def _get_obs(self):
        # combine the cell density and illumination pattern into one observation
        return np.stack([self.state['cell_density'], self.state['illumination_pattern']], axis=0)

    def true_cell_density(self):
        # PLACEHOLDER - replace w empirical data or a simulation model
        return np.random.rand(101, 101)

    def render(self, mode='human'):
        pass
