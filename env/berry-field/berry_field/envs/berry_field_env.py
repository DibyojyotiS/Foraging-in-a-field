import numpy as np
import csv

import gym
from gym import spaces


class BerryFieldEnv(gym.Env):
    def __init__(self,
                 file_paths,
                 num_berries, num_patches,
                 field_size, patch_size, agent_size, observation_space_size,
                 drain_rate, reward_rate,
                 max_steps,
                 initial_state):

        super(BerryFieldEnv, self).__init__()

        # Checking for the correct format of the file_path argument
        if len(file_paths) != 2:
            raise Exception("file_paths should be a list of length 2")

        # Initializing variables
        self.num_berries = num_berries
        self.num_patches = num_patches

        self.field_size = field_size
        self.patch_size = patch_size
        self.agent_size = agent_size

        self.drain_rate = drain_rate
        self.reward_rate = reward_rate

        self.max_steps = max_steps
        self.num_steps = 0

        self.state = initial_state
        self.action_space = spaces.Discrete(9)
        self.observation_space = observation_space_size

        self.action_switcher = {
            1: (0, -1),
            2: (1, -1),
            3: (1, 0),
            4: (1, 1),
            5: (0, 1),
            6: (-1, 1),
            7: (-1, 0),
            8: (-1, -1)
        }

        # Constructing numpy arrays to store the coordinates of berries and patches
        berry_coordinates = np.zeros((self.num_berries, 4), dtype=int)
        patch_coordinates = np.zeros((self.num_patches, 2), dtype=int)

        for file_path, rows in zip(file_paths, [berry_coordinates, patch_coordinates]):
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for i, row in enumerate(csv_reader):
                    rows[i] = np.array([int(float(val)) for val in row], dtype=int)

        assert berry_coordinates.shape == (num_berries, 4)
        assert patch_coordinates.shape == (num_patches, 2)

        # Constructing numpy array to store the current state of the field
        # 1 represents uncollected berry
        # 0 represents empty field
        self.field = np.zeros(field_size, dtype=int)

        # Taking berries to be square
        for berry_info in berry_coordinates:
            # If the dimension of the berry is even, taking the coordinate (given)
            # to be the upper-left corner of the 2X2 square in the center
            if berry_info[1] % 2 == 0:
                left = int(berry_info[1]/2 - 1)
                right = int(berry_info[1]/2)
            # If the dimension of the berry is odd, taking the coordinate (given)
            # to be the center of that square
            else:
                left = int(berry_info[1]/2)
                right = int(berry_info[1]/2)

            self.field[(berry_info[3]-left):(berry_info[3]+right+1),
                       (berry_info[2]-left):(berry_info[2]+right+1)] = berry_info[1]

    # Returns a list of berry sizes hitted by the agent and also removes those berries from the field
    def get_hitted_berries_size(self):
        # If the dimension of the agent is even, taking the coordinate (current state)
        # to be the upper-left corner of the 2X2 square in the center
        if self.agent_size % 2 == 0:
            left = int(self.agent_size / 2 - 1)
            right = int(self.agent_size / 2)
        # If the dimension of the agent is odd, taking the coordinate (current state)
        # to be the center of that square
        else:
            left = int(self.agent_size / 2)
            right = int(self.agent_size / 2)

        covered = self.field[(self.state[1]-left):(self.state[1]+right+1),
                             (self.state[0]-left):(self.state[0]+right+1)]

        hitted_berries_size = []

        # TODO: Fill the hitted_berries_list with the sizes of the hitted berries sizes and return it
        # TODO: Remove the hitted berries from the field

    def get_observation(self):
        pass

    def step(self, action):
        self.num_steps += 1

        done = False
        if self.num_steps == self.max_steps:
            done = True

        reward = -1*self.reward_rate

        move = self.action_switcher.get(action, (0, 0))

        self.state[0] += move[0]
        self.state[1] += move[1]

        hitted_berries_size = get_hitted_berries_size()

        for hitted_berry_size in hitted_berries_size:
            if hitted_berry_size != 0:
                reward += reward_rate*hitted_berry_size

        return get_observation(), reward, done, {}

    def reset(self):
        pass

    def render(self, mode='human'):
        pass