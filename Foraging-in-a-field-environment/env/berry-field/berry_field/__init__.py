from gym.envs.registration import register

register(
    id='berry_field-v0',
    entry_point='berry_field.envs:BerryFieldEnv'
)