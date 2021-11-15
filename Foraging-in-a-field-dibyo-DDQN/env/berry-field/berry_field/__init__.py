from gym.envs.registration import register

# the original environment where the observation space is a MxN image 
# typically M = 1080 and N = 1920
register(
    id='berry_field_original-v0',
    entry_point='berry_field.envs:BerryFieldEnv'
)


# the observation space is a Mx6 matrix in the form [isBerry, direction-vector, distance, size]
register(
    id='berry_field_mat_input-v0',
    entry_point='berry_field.envs:BerryFieldEnv_MatInput'
)