import gym
from constants import *



def get_original_env():
    env = gym.make('berry_field:berry_field_original-v0',
                   file_paths=file_paths,
                   num_berries=num_berries, num_patches=num_patches,
                   field_size=field_size, patch_size=patch_size, agent_size=agent_size,
                   observation_space_size=observation_space_size,
                   drain_rate=drain_rate, reward_rate=reward_rate,
                   max_steps=max_steps,
                   initial_state=initial_state)

    return env

def get_env(observation_type = "ordered", bucket_angle = 45, reward_curiosity = True, 
            reward_curiosity_beta=0.25, reward_grid_size = (100,100)):
    env = gym.make('berry_field:berry_field_mat_input-v0',
                   file_paths=file_paths,
                   num_berries=num_berries, num_patches=num_patches,
                   field_size=field_size, patch_size=patch_size, agent_size=agent_size,
                   observation_space_size=observation_space_size,
                   drain_rate=drain_rate, reward_rate=reward_rate,
                   max_steps=max_steps,
                   initial_state=initial_state,
                   observation_type = observation_type,
                   reward_curiosity = reward_curiosity, 
                   reward_curiosity_beta=reward_curiosity_beta,
                   reward_grid_size = reward_grid_size, # should divide respective dimention of field_size
                   bucket_angle = bucket_angle
                   )

    return env
