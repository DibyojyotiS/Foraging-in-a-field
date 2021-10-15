file_paths = ['data/berry_coordinates.csv', 'data/patch_coordinates.csv']

num_berries = 800
num_patches = 10

field_size = (20000, 20000) # (width, height)
patch_size = (2600, 2600) # (width, height)
agent_size = 10
observation_space_size = (1920, 1080) # (width, height)

drain_rate = 1/(2*120*400)
reward_rate = 1e-4

speed = 400
time = 300
max_steps = speed*time

# initial_state = (9608, 18191)  # Center of 7th patch in the sheet
# initial_state = (10960, 11270) # (width, height)
initial_state = (10100-20, 10650+50)
# initial_state = (8700, 9800)

