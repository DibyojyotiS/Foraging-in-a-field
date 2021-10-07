file_paths = ['data/berry_coordinates.csv', 'data/patch_coordinates.csv']

num_berries = 800
num_patches = 10

field_size = (20000, 20000)
patch_size = (2600, 2600)
agent_size = 10
observation_space_size = (1920, 1080)

drain_rate = 1/(2*120*400)
reward_rate = 1e-4

speed = 400
time = 300
max_steps = speed*time

# initial_state = (18191, 9608)  # Center of 7th patch in the sheet
# initial_state = (9720, 10380)
# initial_state = (9000, 11500)
initial_state = (10000,9950)

