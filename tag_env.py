'''
Defines the gym environment for the tag problem.
'''

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import sys

import pygame
from os import path


# The Warehouse is divided into a grid. Use these 'tiles' to represent the objects on the grid.
class GridTile(Enum):
    _FLOOR=0
    KNUCKLES=1
    SONIC=2

    # Return the first letter of tile name, for printing to the console.
    def __str__(self):
        return self.name[:1]
    

class TagEnv(gym.Env):
    def __init__(self, grid_size=7, fps=1.8):
        super().__init__()
        # Environment dimensions (e.g., a 5x5 grid)
        #replace grid_rows and grid_cols with self.grid_size
        self.grid_size = grid_size
        self.fps = fps
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

        self.last_action=''
        self._init_pygame()

    def _init_pygame(self):
        pygame.init() # initialize pygame
        pygame.display.init() # Initialize the display module

        # Game clock
        self.clock = pygame.time.Clock()

        # Default font
        self.action_font = pygame.font.SysFont("Calibre",30)
        self.action_info_height = self.action_font.get_height()

        # For rendering
        self.cell_height = 64
        self.cell_width = 64
        self.cell_size = (self.cell_width, self.cell_height)        

        # Define game window size (width, height)
        self.window_size = (self.cell_width * self.grid_size, self.cell_height * self.grid_size + self.action_info_height)

        # Initialize game window
        self.window_surface = pygame.display.set_mode(self.window_size) 

        # Load & resize sprites
        file_name = path.join(path.dirname(__file__), "sprites/Sonic.png")
        img = pygame.image.load(file_name)
        self.sonic_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/Knuckles.png")
        img = pygame.image.load(file_name)
        self.knuckles_img = pygame.transform.scale(img, self.cell_size)

        file_name = path.join(path.dirname(__file__), "sprites/floor.png")
        img = pygame.image.load(file_name)
        self.floor_img = pygame.transform.scale(img, self.cell_size)


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

        return (tagger_obs, runner_obs), 0, done, truncated, {'tagger_reward':tagger_reward, 'runner_reward':runner_reward}

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

    def render2(self):
        # Render the grid (optional, useful for debugging)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = "."
        grid[self.runner_pos[1], self.runner_pos[0]] = "R"
        grid[self.tagger_pos[1], self.tagger_pos[0]] = "T"
        print("\n".join([" ".join(row) for row in grid]))
        print("------------------------------------")

    def render(self):
        # Print current state on console
        for r in range(self.grid_size):
            for c in range(self.grid_size):

                if([r,c] == list(self.tagger_pos)):
                    print(GridTile.KNUCKLES, end=' ')
                elif([r,c] == list(self.runner_pos)):
                    print(GridTile.SONIC, end=' ')
                else:
                    print(GridTile._FLOOR, end=' ')

            print() # new line
        print() # new line

        self._process_events()

        # clear to white background, otherwise text with varying length will leave behind prior rendered portions
        self.window_surface.fill((255,255,255))

        # Print current state on console
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                
                # Draw floor
                pos = (c * self.cell_width, r * self.cell_height)
                self.window_surface.blit(self.floor_img, pos)

                if([r,c] == list(self.runner_pos)):
                    # Draw robot
                    self.window_surface.blit(self.sonic_img, pos)
                if([r,c] == list(self.tagger_pos)):
                    # Draw robot
                    self.window_surface.blit(self.knuckles_img, pos)
                
        text_img = self.action_font.render(f'Action: {self.last_action}', True, (0,0,0), (255,255,255))
        text_pos = (0, self.window_size[1] - self.action_info_height)
        self.window_surface.blit(text_img, text_pos)       

        pygame.display.update()
                
        # Limit frames per second
        self.clock.tick(self.fps)  


    def _process_events(self):
        # Process user events, key presses
        for event in pygame.event.get():
            # User clicked on X at the top right corner of window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if(event.type == pygame.KEYDOWN):
                # User hit escape
                if(event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

class Move(Enum):
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3
    UP_LEFT=4
    UP_RIGHT=5
    DOWN_LEFT=6
    DOWN_RIGHT=7