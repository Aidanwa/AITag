'''
Defines the gym environment for the tag problem.
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum

class TagEnv(gym.Env):
    def __init__(self, grid_size=7):
        super().__init__()
        # Environment dimensions (e.g., a 5x5 grid)
        self.grid_size = grid_size
        # Action space: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Tuple((
            spaces.Discrete(8),  # Tagger's action
            spaces.Discrete(8)   # Runner's action
        ))

        # Observation space for each agent: (tagger_x, tagger_y, runner_x, runner_y)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32),  # Tagger's observation
            spaces.Box(low=0, high=grid_size - 1, shape=(4,), dtype=np.int32)   # Runner's observation
        ))

        self.reset()
    
    def reset(self, seed=None, options=None):
        # Initialize positions
        self.tagger_pos = np.array([0, 0])  # Tagger starts at top-left
        self.runner_pos = np.array([self.grid_size - 1, self.grid_size - 1])  # Runner starts at bottom-right

        # Return initial observations
        tagger_obs = np.concatenate([self.tagger_pos, self.runner_pos])
        runner_obs = np.concatenate([self.runner_pos, self.tagger_pos])

        return (tagger_obs, runner_obs), {}

    def step(self, actions):

        tagger_action, runner_action = actions

        # Update positions
        tagger_valid = self._move(self.tagger_pos, tagger_action)
        runner_valid = self._move(self.runner_pos, runner_action)

        # Check if the tagger catches the runner
        caught = np.array_equal(self.tagger_pos, self.runner_pos)

        # Calculate rewards
        tagger_reward = 10 if caught else -1 + tagger_valid # Tagger gets +10 for catching, -1 otherwise
        runner_reward = -10 if caught else 1 + runner_valid # Runner gets -10 for getting caught, +1 otherwise

        # Create observations
        tagger_obs = np.concatenate([self.tagger_pos, self.runner_pos])
        runner_obs = np.concatenate([self.runner_pos, self.tagger_pos])

        done = caught  # Episode ends if tagger catches runner
        truncated = False

        return (tagger_obs, runner_obs), 0, done, _, {'tagger_reward':tagger_reward, 'runner_reward':runner_reward}

    def _move(self, position, action):
        # Move within bounds
        if action == 0 and position[1] > 0:  # Move up (decrease y-coordinate)
            position[1] -= 1
        elif action == 1 and position[1] < self.grid_size - 1:  # Move down (increase y-coordinate)
            position[1] += 1
        elif action == 2 and position[0] > 0:  # Move left (decrease x-coordinate)
            position[0] -= 1
        elif action == 3 and position[0] < self.grid_size - 1:  # Move right (increase x-coordinate)
            position[0] += 1
        elif action == 4 and position[0] > 0 and position[1] > 0:  # Move top-left (decrease x and y coordinates)
            position[0] -= 1
            position[1] -= 1
        elif action == 5 and position[0] < self.grid_size - 1 and position[1] > 0:  # Move top-right (increase x, decrease y)
            position[0] += 1
            position[1] -= 1
        elif action == 6 and position[0] > 0 and position[1] < self.grid_size - 1:  # Move bottom-left (decrease x, increase y)
            position[0] -= 1
            position[1] += 1
        elif action == 7 and position[0] < self.grid_size - 1 and position[1] < self.grid_size - 1:  # Move bottom-right (increase x and y)
            position[0] += 1
            position[1] += 1
        else: 
            return -1
        return 0

    def render(self):
        # Render the grid (optional, useful for debugging)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = "."
        grid[self.runner_pos[1], self.runner_pos[0]] = "R"
        grid[self.tagger_pos[1], self.tagger_pos[0]] = "T"
        print("\n".join([" ".join(row) for row in grid]))
        print("------------------------------------")

class Move(Enum):
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3
    UP_LEFT=4
    UP_RIGHT=5
    DOWN_LEFT=6
    DOWN_RIGHT=7