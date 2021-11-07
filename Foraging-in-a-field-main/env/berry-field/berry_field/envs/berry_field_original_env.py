import numpy as np
import csv
import gym
import pyglet
from matplotlib import pyplot as plt
import pygame
from pygame import surfarray
from .utils.interval_tree import IntervalTree


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

        self.state = [initial_state[0] + observation_space_size[0]//2, initial_state[1] + observation_space_size[1]//2]
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = observation_space_size

        # initializing variables Pygame render
        self.display = None
        self.image = None
        self.font = None
        self.cummulative_reward = 0.5 # initial reward is 0.5
        self.last_action = None


        self.action_switcher = {
            7: (0, -1),
            6: (1, -1),
            5: (1, 0),
            4: (1, 1),
            3: (0, 1),
            2: (-1, 1),
            1: (-1, 0),
            8: (-1, -1)
        }

        # Constructing numpy array to store the current state of the field
        # 1 represents uncollected berry
        # 0 represents empty field
        padded_field_size = (field_size[0] + observation_space_size[0], field_size[1] + observation_space_size[1])
        self.field = np.zeros(padded_field_size, dtype=int)

        self.boundary_val = -1
        self.field[observation_space_size[0]//2:observation_space_size[0]//2+field_size[0]+1, observation_space_size[1]//2] = self.boundary_val
        self.field[observation_space_size[0]//2, observation_space_size[1]//2:observation_space_size[1]//2+field_size[1]+1] = self.boundary_val
        self.field[observation_space_size[0]//2:observation_space_size[0]//2+field_size[0]+1, observation_space_size[1]//2 + field_size[1]] = self.boundary_val
        self.field[observation_space_size[0]//2 + field_size[0], observation_space_size[1]//2:observation_space_size[1]//2+field_size[1]+1] = self.boundary_val


        # Constructing numpy arrays to store the coordinates of berries and patches
        berry_coordinates = np.zeros((self.num_berries, 4), dtype=int)
        patch_coordinates = np.zeros((self.num_patches, 2), dtype=int)

        for file_path, rows in zip(file_paths, [berry_coordinates, patch_coordinates]):
            with open(file_path, 'r') as file:
                csv_reader = csv.reader(file)
                for i, row in enumerate(csv_reader):
                    rows[i] = np.array([int(float(val)) for val in row], dtype=int)
        
        # account for padding in the berry and patch coordinates
        berry_coordinates[:,2] += observation_space_size[0]//2
        berry_coordinates[:,3] += observation_space_size[1]//2
        patch_coordinates[:,0] += observation_space_size[0]//2
        patch_coordinates[:,1] += observation_space_size[1]//2


        assert berry_coordinates.shape == (self.num_berries, 4)
        assert patch_coordinates.shape == (self.num_patches, 2)

        self.berry_info = berry_coordinates.copy()  # This is used to match berry_id(index) to their size and position
        self.berry_info[:, 0] = 1
        data_h = []  # To construct interval tree in the X-direction
        data_w = []  # To construct interval tree in the Y-direction


        # Taking berries to be square
        for berry_id, berry_info in enumerate(berry_coordinates):
            left, right, _, _ = self.get_range(berry_info[1], berry_info[1])
            data_h.append([berry_info[2]-left, berry_info[2]+right, berry_id])
            data_w.append([berry_info[3]-left, berry_info[3]+right, berry_id])
            self.field[(berry_info[2]-left):(berry_info[2]+right+1),
                       (berry_info[3]-left):(berry_info[3]+right+1)] += berry_info[1]

        # # Constructing interval trees for X (vertical) and Y(horizontal) direction
        self.interval_tree_h = IntervalTree(data_h)
        self.interval_tree_w = IntervalTree(data_w)



    def get_range(self, size_h, size_w):
        # If the size is even, taking the coordinate (given) to be the upper-left corner of the 2X2 square in the center
        # If the size is odd, taking the coordinate (given) to be the center of that square

        if size_w % 2 == 0:
            left = int(size_w / 2 - 1)
        else:
            left = int(size_w / 2)
        right = int(size_w / 2)

        if size_h % 2 == 0:
            up = int(size_h / 2 - 1)
        else:
            up = int(size_h / 2)
        down = int(size_h / 2)

        return left, right, up, down
        

    # Removes berry from the field
    def remove_berry(self, berry_id):
        self.berry_info[berry_id, 0] = 0
        berry_info = self.berry_info[berry_id]
        left, right, _, _ = self.get_range(berry_info[1], berry_info[1])
        self.field[(berry_info[2] - left):(berry_info[2] + right + 1),
                   (berry_info[3] - left):(berry_info[3] + right + 1)] -= berry_info[1]


    # Returns a list of berry sizes hitted by the agent and also removes those berries from the field
    def get_hitted_berries_size(self):
        left, right, up, down = self.get_range(self.agent_size, self.agent_size)

        covered = self.field[(self.state[0]-left):(self.state[0]+right+1),
                             (self.state[1]-left):(self.state[1]+right+1)]

        if np.sum(covered) <=0: return []

        berrys_along_vertical = self.interval_tree_h.find_overlaps((self.state[0] - up, self.state[0] + down))
        berrys_along_horizontal = self.interval_tree_w.find_overlaps((self.state[1] - left, self.state[1] + right))
        hitted_berries_id = berrys_along_vertical.intersection(berrys_along_horizontal)

        print(
            # (self.state[0] - up, self.state[0] + down), '\n',
            # berrys_along_vertical, '\n',
            # (self.state[1] - left, self.state[1] + right), '\n',
            # berrys_along_horizontal, '\n',
            hitted_berries_id, '\n\n'
        )

        hitted_berries_size = []
        for berry_id in hitted_berries_id:
            if self.berry_info[berry_id, 0] == 1:
                self.remove_berry(berry_id)
                hitted_berries_size.append(self.berry_info[1,1])

        return hitted_berries_size  

    def get_observablefield(self):
        left, right, up, down = self.get_range(self.observation_space[0], self.observation_space[1])
        H1 = max(0, self.state[0] - up)
        H2 = min(self.state[0] + down+1, self.observation_space[0] + self.field_size[0]+1)
        W1 = max(0, self.state[1] - left)
        W2 = min(self.state[1] + right+1, self.observation_space[1] + self.field_size[1]+1)
        # print((H1,H2), (W1,W2), self.observation_space[0] + self.field_size[0]+1, self.state[0] + down+1)
        return self.field[H1:H2, W1:W2]

    def get_observation(self):
        return (self.get_observablefield() > 0).astype('uint8')


    def step(self, action):
        self.num_steps += 1

        done = False
        if self.num_steps == self.max_steps:
            done = True

        reward = -1*self.drain_rate

        move = self.action_switcher.get(action, (0, 0))

        self.state[0] = min(max(self.observation_space[0]//2+self.agent_size//2, self.state[0] + move[0]), self.observation_space[0]//2+self.field_size[0]-self.agent_size//2)
        self.state[1] = min(max(self.observation_space[1]//2+self.agent_size//2, self.state[1] + move[1]), self.observation_space[1]//2+self.field_size[1]-self.agent_size//2)

        hitted_berries_size = self.get_hitted_berries_size()

        for hitted_berry_size in hitted_berries_size:
            if hitted_berry_size != 0:
                reward += self.reward_rate*hitted_berry_size

        self.cummulative_reward += reward
        self.last_action = action

        return self.get_observation(), reward, done, {}


    def reset(self):
        pass


    def render(self, mode='human'):

        if self.display is None:
            pygame.init()
            self.font = pygame.font.Font(None, 32)
            self.image = np.zeros((self.observation_space[1], self.observation_space[0], 3), dtype=np.uint8) # place-holder array
            self.display = pygame.display.set_mode((self.observation_space[1], self.observation_space[0]))

        view = self.get_observablefield()
        layer1 = (view > 0).astype('uint8') * 255
        layer2 = (view == 0).astype('uint8') * 255
        layer3 = (view == self.boundary_val).astype('uint8') * 255
        img = np.dstack((layer1, layer2, layer3))

        self.image[:][:][:] = np.transpose(img, [1,0,2])

        # draw agent
        left, right, _, _ = self.get_range(self.agent_size, self.agent_size)
        left_w, right_w, up_w, down_w = self.get_range(self.observation_space[0], self.observation_space[1])
        self.image[(left_w - left):(left_w + right + 1), (up_w - left):(up_w + right + 1), :] = 255


        text = self.font.render(f'cummulative-reward: {self.cummulative_reward:.3f}   pos: {self.state}   action: {self.last_action}', (0, 255, 0), (0,0,0))
        textRect = text.get_rect()
        textRect.center = (int(0.3 * self.observation_space[1]), int(0.15 * self.observation_space[0]))

        surfarray.blit_array(self.display, self.image)
        self.display.blit(text, textRect)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.display = self.image = None
                pygame.quit()



    def getnumpyhumanrender(self):
        view = self.get_observablefield()
        layer1 = (view > 0).astype('uint8') * 255
        layer2 = (view == 0).astype('uint8') * 255
        layer3 = (view == self.boundary_val).astype('uint8') * 255
        img = np.dstack((layer1, layer2, layer3))
        left, right, _, _ = self.get_range(self.agent_size, self.agent_size)
        left_w, right_w, up_w, down_w = self.get_range(self.observation_space[0], self.observation_space[1])
        img[(up_w - left):(up_w + right + 1), (left_w - left):(left_w + right + 1), :] = 255
        # img = np.transpose(img, [1,0,2])
        return img


    def showfield(self):
        view = self.field
        layer1 = (view > 0).astype('uint8') * 255
        layer2 = (view == 0).astype('uint8') * 255
        layer3 = (view == self.boundary_val).astype('uint8') * 255
        img = np.dstack((layer1, layer2, layer3))
        plt.imshow(img)
        plt.show()