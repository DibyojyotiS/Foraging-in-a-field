import gym

from constants import *


def get_env():
    env = gym.make('berry_field:berry_field-v0',
                   file_paths=file_paths,
                   num_berries=num_berries, num_patches=num_patches,
                   field_size=field_size, patch_size=patch_size, agent_size=agent_size,
                   observation_space_size=observation_space_size,
                   drain_rate=drain_rate, reward_rate=reward_rate,
                   max_steps=max_steps,
                   initial_state=initial_state)

    return env
