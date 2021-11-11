import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import get_env


env = get_env.get_env()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def collectExperiences(self, env, state, numExperiences, toRender):
        for i in range(numExperiences):
            # printEps = True if i % 1000 == 0 or i == 19999 else False
            printEps = False
            action = select_action(state, printEps)

            next_state, reward, done, _ = env.step(action)

            next_state = next_state.flatten()
            next_state = torch.from_numpy(next_state)
            reward = torch.tensor([reward], device=None)

            memory.push(state, action, next_state, reward)

            state = next_state

            if toRender:
                env.render()

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Linear(input_size, 16)
        self.conv2 = nn.Linear(16, 32)
        self.conv3 = nn.Linear(32, output_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 4000
TARGET_UPDATE = 10
action_space_size = 9
input_size = 35*5

n_actions = action_space_size

policy_net = DQN(input_size, n_actions).to(dtype=float)
target_net = DQN(input_size, n_actions).to(dtype=float)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), 0.002)
memory = ReplayMemory(20000)


steps_done = 0


def getBestAction(state):
    minDistVector = [0, 0]
    minDist = 1e10
    for i in range(35):
        if state[i*5 + 3] < minDist and state[i*5 + 3] != 0:
            minDistVector[0] = state[i*5 + 1]
            minDistVector[1] = -state[i*5 + 2]
            minDist = state[i*5 + 3]

    if minDistVector[0] == 0:
        return 1
    else:
        tanTheta = minDistVector[1]/minDistVector[0]
        if minDistVector[0] > 0:
            angle = np.arctan(tanTheta)*180/np.pi
            if angle < -67.5:
                return 1
            if angle < -22.5:
                return 2
            if angle < 22.5:
                return 3
            if angle < 67.5:
                return 4
            if angle < 90:
                return 5
        else:
            angle = 180 + np.arctan(tanTheta)*180/np.pi
            if angle < 112.5:
                return 5
            if angle < 157.5:
                return 6
            if angle < 202.5:
                return 7
            if angle < 247.5:
                return 8
            if angle < 270:
                return 1


def select_action(state, printEps):
    global steps_done
    sample = random.random()

    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if printEps:
        print(eps_threshold)

    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).argmax()
    else:
        if random.random() < 0.4:
            return torch.tensor(np.random.randint(9, size=1)[0])
        else:
            actionForClosestBerry = getBestAction(state)
            return torch.tensor(actionForClosestBerry)
        # return torch.tensor(np.random.randint(9, size=1)[0])


episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=None, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(action_batch.shape)
    state_action_values = policy_net(state_batch).gather(
        1, action_batch.reshape(128, 1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=None, dtype=float)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (
        next_state_values * GAMMA) + reward_batch.squeeze(1)

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values,
                     expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    steps_done = 0
    EPS_START *= 0.9
    if EPS_START < EPS_END:
        EPS_START = EPS_END

    state, done = env.reset()
    state = state.flatten()
    state = torch.from_numpy(state)

    toRender = False
    memory.collectExperiences(env, state, 20000, toRender)

    for t in range(180):
        sample = memory.sample(BATCH_SIZE)

        optimize_model()

    print(
        f'Episode = {i_episode} done, cumulativeReward = {env.cummulative_reward}')

    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
torch.save(policy_net.state_dict(), 'trainedModel.pth')
env.render()
plt.ioff()
plt.show()
