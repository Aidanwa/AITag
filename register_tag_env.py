'''
script to register environment
'''

from gymnasium.envs.registration import register

register(
    id="Tag-v0",  # Unique name for the environment
    entry_point="tag_env:TagEnv",  # Path to the environment class
    max_episode_steps=200
)