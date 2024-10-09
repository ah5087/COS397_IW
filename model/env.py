import gym
import numpy as np
from torch.utils.data import DataLoader
from dataset import CellMigrationDataset

class CellMigrationEnv(gym.Env):
    def __init__(self, data_loader):
        super(CellMigrationEnv, self).__init__()
        self.data_loader = data_loader
        self.current_step = 0

        # Observation space: Assume each observation is a 2D grid of size 101x101 (cell density map)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(2, 101, 101), dtype=np.float32)
        
        # Action space: Now, the action will represent a 2D predicted density map
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(101, 101), dtype=np.float32)  # 2D density map

        self.data_iterator = iter(self.data_loader)  # Create an iterator for the data

    def reset(self):
        # Reset environment at the start of an episode
        self.current_step = 0
        try:
            # Reset to the next sequence in the data (current ground truth)
            self.state, _ = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)  # Restart the iterator
            self.state, _ = next(self.data_iterator)

        return self.state

    def step(self, predicted_density):
        """
        Args:
            predicted_density: A 2D grid of predicted cell densities, shape: (101, 101)
        """
        self.current_step += 1

        # Fetch the next state (next frame of the time-lapse video)
        try:
            next_state, _ = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)  # Restart the iterator
            next_state, _ = next(self.data_iterator)

        # Define your reward function: e.g., negative mean squared error between predicted density and current state
        reward = -np.mean((predicted_density - self.state[0].cpu().numpy()) ** 2)

        # Define your done condition (you can modify this as needed)
        done = self.current_step >= 100  # End episode after 100 steps

        # Set next state as the current state
        self.state = next_state

        # Return next_state, reward, done, and an empty info dictionary
        return next_state, reward, done, {}
