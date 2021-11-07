import numpy as np
import csv
import gym
import pyglet
from matplotlib import pyplot as plt
import pygame
from pygame import surfarray


from .interval_tree import IntervalTree
from .rendering import Viewer


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

        self.state = list(initial_state)
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = observation_space_size

        self.display = None
        self.image = None

        self.window = None

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

        assert berry_coordinates.shape == (self.num_berries, 4)
        assert patch_coordinates.shape == (self.num_patches, 2)

        self.berry_info = berry_coordinates.copy()  # This is used to match berry_id(index) to their size and position
        self.berry_info[:, 0] = 1
        data_x = []  # To construct interval tree in the X-direction
        data_y = []  # To construct interval tree in the Y-direction

        # Constructing numpy array to store the current state of the field
        # 1 represents uncollected berry
        # 0 represents empty field
        self.field = np.zeros(field_size, dtype=int)

        # Taking berries to be square
        for berry_id, berry_info in enumerate(berry_coordinates):
            left, right, _, _ = self.get_range(berry_info[1], berry_info[1])

            data_x.append([berry_info[2]-left, berry_info[2]+right, berry_id])
            data_y.append([berry_info[3]-left, berry_info[3]+right, berry_id])

            self.field[(berry_info[3]-left):(berry_info[3]+right+1),
                       (berry_info[2]-left):(berry_info[2]+right+1)] += berry_info[1]

        # Constructing interval trees for X and Y direction
        self.interval_tree_x = IntervalTree(data_x, 0, field_size[0])
        self.interval_tree_y = IntervalTree(data_y, 0, field_size[0])

    def get_range(self, size_x, size_y):
        # If the size is even, taking the coordinate (given) to be the upper-left corner of the 2X2 square in the center
        # If the size is odd, taking the coordinate (given) to be the center of that square

        if size_x % 2 == 0:
            left = int(size_x / 2 - 1)
        else:
            left = int(size_x / 2)
        right = int(size_x / 2)

        if size_y % 2 == 0:
            up = int(size_y / 2 - 1)
        else:
            up = int(size_y / 2)
        down = int(size_y / 2)

        return left, right, up, down

    # Removes berry from the field
    def remove_berry(self, berry_id):
        print('remove')
        self.berry_info[berry_id, 0] = 0

        berry_info = self.berry_info[berry_id]

        left, right, _, _ = self.get_range(berry_info[1], berry_info[1])

        self.field[(berry_info[3] - left):(berry_info[3] + right + 1),
                   (berry_info[2] - left):(berry_info[2] + right + 1)] -= berry_info[1]

    # Returns a list of berry sizes hitted by the agent and also removes those berries from the field
    def get_hitted_berries_size(self):
        left, right, _, _ = self.get_range(self.agent_size, self.agent_size)

        covered = self.field[(self.state[1]-left):(self.state[1]+right+1),
                             (self.state[0]-left):(self.state[0]+right+1)]

        hitted_berries_size = []
        hitted_berries_id = set()

        berry_ids_y = self.interval_tree_y.find_intervals(self.state[1] - left)
        for p in range(self.state[0] - left, self.state[0] + right + 1):
            berry_ids_x = self.interval_tree_x.find_intervals(p)
            hitted_berries_id.update(berry_ids_x & berry_ids_y)

        berry_ids_y = self.interval_tree_y.find_intervals(self.state[1] + right)
        for p in range(self.state[0] - left, self.state[0] + right + 1):
            berry_ids_x = self.interval_tree_x.find_intervals(p)
            hitted_berries_id.update(berry_ids_x & berry_ids_y)

        berry_ids_x = self.interval_tree_x.find_intervals(self.state[0] - left)
        for p in range(self.state[1] - left + 1, self.state[1] + right):
            berry_ids_y = self.interval_tree_y.find_intervals(p)
            hitted_berries_id.update(berry_ids_x & berry_ids_y)

        berry_ids_x = self.interval_tree_x.find_intervals(self.state[0] + right)
        for p in range(self.state[1] - left + 1, self.state[1] + right):
            berry_ids_y = self.interval_tree_y.find_intervals(p)
            hitted_berries_id.update(berry_ids_x & berry_ids_y)

        #print(hitted_berries_id)

        for berry_id in hitted_berries_id:
            if self.berry_info[berry_id, 0] == 1:
                self.remove_berry(berry_id)
                hitted_berries_size.append(self.berry_info[1,1])

        return hitted_berries_size

    def get_observation(self):
        left, right, up, down = self.get_range(self.observation_space[0], self.observation_space[1])

        if self.state[0] < left:
            right = right + left - self.state[0]
            left = self.state[0]
        if self.state[1] < up:
            down = down + up - self.state[1]
            up = self.state[1]
        if self.field_size[0] - self.state[0] - 1 < right:
            left = left + right - self.field_size[0] + self.state[0] + 1
            right = self.field_size[0] - self.state[0] - 1
        if self.field_size[1] - self.state[1] - 1 < down:
            up = up + down - self.field_size[1] + self.state[1] + 1
            down = self.field_size[1] - self.state[1] - 1

        return self.field[(self.state[1] - up):(self.state[1] + down + 1),
                          (self.state[0] - left):(self.state[0] + right + 1)]

        # return self.field[(self.state[1]-(int)(self.observation_space[1]/2)):(self.state[1]+(int)(self.observation_space[1]/2)),
        #        (self.state[0]-(int)(self.observation_space[0]/2)):((int)(self.state[0]+self.observation_space[0]/2))]

    def step(self, action):
        self.num_steps += 1

        done = False
        if self.num_steps == self.max_steps:
            done = True

        reward = -1*self.drain_rate

        move = self.action_switcher.get(action, (0, 0))

        self.state[0] += move[0]
        self.state[1] += move[1]

        hitted_berries_size = self.get_hitted_berries_size()

        for hitted_berry_size in hitted_berries_size:
            if hitted_berry_size != 0:
                reward += self.reward_rate*hitted_berry_size

        return self.get_observation(), reward, done, {}

    def reset(self):
        pass

    def render(self, mode='human'):

        if self.display is None:
            # self.display = Viewer(self.observation_space[0], self.observation_space[1], 'Display Window', resizable=True)
            pygame.init()
            self.image = np.zeros((1920, 1080, 3), dtype=np.uint8)
            self.display = pygame.display.set_mode((1920, 1080))

        view = self.get_observation()
        ax, ay = self.state

        layer1 = (view > 0).astype('uint8') * 255
        layer2 = (view == 0).astype('uint8') * 255
        layer3 = np.zeros_like(layer2, dtype=np.uint8)
        img = np.dstack((layer1, layer2, layer3))

        self.image[:][:][:] = np.transpose(img, [1,0,2])

        left, right, _, _ = self.get_range(self.agent_size, self.agent_size)
        left_w, right_w, up_w, down_w = self.get_range(self.observation_space[0], self.observation_space[1])
        self.image[(left_w - left):(left_w + right + 1), (up_w - left):(up_w + right + 1), :] = 255

        # self.display.draw_image(self.image)
        # pyglet.app.run()

        surfarray.blit_array(self.display, self.image)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.display = self.image = None
                pygame.quit()

    # def render(self, mode='human'):
    #     view = self.get_observation()
    #
    #     #print(np.sum(view))
    #     #print(view.shape)
    #
    #     layer1 = (view > 0).astype('uint8') * 255
    #     layer2 = (view == 0).astype('uint8') * 255
    #     layer3 = np.zeros((self.observation_space[1], self.observation_space[0]), dtype='uint8')
    #
    #     image = np.dstack((layer1, layer2, layer3))
    #
    #     left, right, _, _ = self.get_range(self.agent_size, self.agent_size)
    #     left_w, right_w, up_w, down_w = self.get_range(self.observation_space[0], self.observation_space[1])
    #
    #     image[(up_w - left):(up_w + right + 1),
    #           (left_w - left):(left_w + right + 1),
    #           :] = 0
    #
    #     plt.imshow(image)
    #     plt.show()

        # if self.window is None:
        #     self.window = Viewer(self.observation_space[0], self.observation_space[1], 'Display Window', resizable=True)
        #
        # self.window.draw_image(image)
        # pyglet.app.run()
