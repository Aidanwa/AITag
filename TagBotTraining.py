'''
Setting up the RL training for both Q learning and Stable Baselines 3
'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
# from stable_baselines3 import A2C
import os
import register_tag_env
from tag_env import Move

def runQLearning(episodes, is_training=True, render=False, random_runner=False, continue_training=False, runner_training=False, tagger_training=False):

    env = gym.make('Tag-v0')
    grid_size = env.unwrapped.grid_size

    if is_training and not continue_training:
        # If training, initialize the Q Table, a 5D vector: [robot_row_pos, robot_row_col, target_row_pos, target_col_pos, actions]
        
        if tagger_training:
            with open('tagger_q_table.pkl', 'rb') as f:
                q_runner = pickle.load(f)
        elif not random_runner:
            q_runner = np.zeros((grid_size, grid_size,grid_size, grid_size, env.action_space[1].n))
        if runner_training:
            with open('tagger_q_table.pkl', 'rb') as f:
                q_tagger = pickle.load(f)
        else:
            q_tagger = np.zeros((grid_size, grid_size, grid_size, grid_size, env.action_space[0].n))

    else:
        # Load Q-Tables from file
        if not random_runner:
            with open('runner_q_table.pkl', 'rb') as f:
                q_runner = pickle.load(f)
        with open('tagger_q_table.pkl', 'rb') as f:
            q_tagger = pickle.load(f)

    # Hyperparameters
    learning_rate = 0.7   # alpha or learning rate
    discount_factor = 0.8 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1             # 1 = 100% random actions

    if not is_training:
        epsilon = 0

    # Array to keep track of the number of steps per episode for the robot to find the target.
    # We know that the robot will inevitably find the target, so the reward is always obtained,
    # so we want to know if the robot is reaching the target efficiently.
    steps_per_episode = np.zeros(episodes)
    step_count=0

    for i in range(episodes):
        if render:
            print(f'Episode {i}')

        # Reset environment at the beginning of episode
        (tagger_state, runner_state), _ = env.reset()
        terminated, truncated = False, False

        # Robot keeps going until it finds the target
        while(not terminated and not truncated):
            if render:
                print(f"Turn {step_count}")

            valid_tagger_actions = get_valid_actions(tagger_state[0:2], grid_size)
            valid_runner_actions = get_valid_actions(runner_state[0:2], grid_size)

            # Select action based on epsilon-greedy
            if is_training and random.random() < epsilon:
                action_tagger = random.choice(valid_tagger_actions)
            else:
                q_tagger_idx = tuple(tagger_state)
                action_tagger = np.argmax(q_tagger[q_tagger_idx])

            if not random_runner:
                # Runner selects an action
                if is_training and random.random() < epsilon:
                    action_runner = random.choice(valid_runner_actions)
                else:
                    q_runner_idx = tuple(runner_state)
                    action_runner = np.argmax(q_runner[q_runner_idx])
            else:
                action_runner = random.choice(valid_runner_actions)

            # Perform action
            new_states, _, terminated, truncated, info = env.step((action_tagger, action_runner))
            tagger_new_state, runner_new_state = new_states
            tagger_reward, runner_reward = info['tagger_reward'], info['runner_reward']
            if render:
                print(f"Tagger move:{Move(action_tagger)}, Runner move: {Move(action_runner)}")
                env.render()

            
            if not runner_training:
                # Update Q-Table for tagger
                q_tagger_state_action_idx = tuple(tagger_state) + (action_tagger,)
                q_tagger_new_state_idx = tuple(tagger_new_state)
                if is_training:
                    q_tagger[q_tagger_state_action_idx] += learning_rate * (
                        tagger_reward + discount_factor * np.max(q_tagger[q_tagger_new_state_idx]) - q_tagger[q_tagger_state_action_idx]
                    )
            if not random_runner and not tagger_training:
                # Update Q-Table for runner
                q_runner_state_action_idx = tuple(runner_state) + (action_runner,)
                q_runner_new_state_idx = tuple(runner_new_state)
                if is_training:
                    q_runner[q_runner_state_action_idx] += learning_rate * (
                        runner_reward + discount_factor * np.max(q_runner[q_runner_new_state_idx]) - q_runner[q_runner_state_action_idx]
                    )

            # Update current state
            tagger_state, runner_state = tagger_new_state, runner_new_state

            # Record steps
            step_count+=1
            if terminated or truncated:
                steps_per_episode[i] = step_count
                step_count = 0

        # Decrease epsilon
        if is_training:
            epsilon = max(1 - (i / (0.5 * episodes)), 0.01)  # Faster initial decay
        if render:
            print(f"Finished {i} episodes")
        if i % 100 == 0:
            print(f'Finished {i} episodes', end='\r')

    # Graph steps
    if is_training:
        print(f"Finished {episodes} episodes")
        sum_steps = np.zeros(episodes)

        for t in range(episodes):
            sum_steps[t] = np.mean(steps_per_episode[max(0, t - 100):(t + 1)])  # Average steps over 100 episodes
        plt.figure(figsize=(10, 6))  # Set a larger figure size
        plt.plot(sum_steps, label='Avg Num Steps over 100 Episodes', color='b')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Avg Num Steps over 100 Episodes', fontsize=12)
        plt.title('Training Progress: Avg Num Steps over 100 Episodes', fontsize=14)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('tag_env_training.png')

        # Save Q-Tables
        if not random_runner and not tagger_training:
            with open("runner_q_table.pkl", "wb") as f:
                pickle.dump(q_runner, f)
        if not runner_training:
            with open("tagger_q_table.pkl", "wb") as f:
                pickle.dump(q_tagger, f)


def get_valid_actions(agent_pos, grid_size):
    valid_actions = []
    x, y = agent_pos

    moves = {
        Move.UP: (x, y - 1),
        Move.DOWN: (x, y + 1),
        Move.LEFT: (x - 1, y),
        Move.RIGHT: (x + 1, y),
        Move.UP_LEFT: (x - 1, y - 1),
        Move.UP_RIGHT: (x + 1, y - 1),
        Move.DOWN_LEFT: (x - 1, y + 1),
        Move.DOWN_RIGHT: (x + 1, y + 1)
    }

    for action, (new_row, new_col) in moves.items():
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            valid_actions.append(action.value)

    return valid_actions

if __name__ == "__main__":
    ''' Training and Testing the Qlearning of the Tag Bot '''
    # runQLearning(episodes=50000, is_training=True, render=False, random_runner=True)
    # runQLearning(5, is_training=False, render=True, random_runner=True)

    ''' Training a Q learner bot using the trained TagBot'''
    # runQLearning(episodes=10000, is_training=True, render = False, runner_training=True)
    # runQLearning(2, is_training=False, render=True)

    ''' Training and Testing the Qlearning of the Tag Bot '''
    # runQLearning(episodes=50000, is_training=True, render=False, tagger_training=True)
    # runQLearning(2, render=True,is_training=False,continue_training=False)

    ''' using both learning agents '''
    # runQLearning(episodes=100000, is_training=True, render = False)
    # runQLearning(1, is_training=False, render=True)

    ''' seeing how the fully advanced tagger does against a random runner. '''
    # runQLearning(5, is_training=False, random_runner=True, render=True)

    ''' final test of initial, training tagger, then runner, then both'''
    runQLearning(episodes=50000, is_training=True, render=False, random_runner=True)
    runQLearning(5, is_training=False, render=True, random_runner=True)
    runQLearning(episodes=10000, is_training=True, render = False, runner_training=True)
    runQLearning(2, is_training=False, render=True)
    runQLearning(episodes=500000, is_training=True, render = False)
    runQLearning(1, is_training=False, render=True)

