import os

import gym
import math
import random
import numpy as np
import matplotlib
import get_env
import matplotlib.pyplot as plt
from itertools import count
from PIL import Image

# from priority_experience_replay import Memory

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = get_env.get_env(observation_type="buckets")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.inputlayer = torch.nn.Linear(18, 32, bias=True)
        self.hl1 = torch.nn.Linear(32, 8, bias=True)
        self.hl2 = torch.nn.Linear(8, 8, bias=True)
        self.outlayer = torch.nn.Linear(8, 9, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        x = x.view(x.size(0), -1)  # flatten but conserve batches
        x = F.relu(self.inputlayer(x))
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = self.outlayer(x)
        return x


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        assert len(sample) == 5
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = np.zeros(n)

        self.beta = min(1., self.beta + self.beta_increment_per_sampling)

        i = 0
        while i < n:
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            if data is None:
                continue
            priorities[i] = p
            batch.append(data)
            idxs.append(idx)
            i += 1

        sampling_probabilities = priorities / \
            self.tree.total() + 10E-8  # for zero priority events
        is_weight = np.power(self.tree.n_entries *
                             sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None for _ in range(capacity)]
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        assert len(data) == 5
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 50
TARGET_UPDATE = 2

policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)
memory = Memory(20000)


def select_action(state, episode):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episode / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(9)]], device=device, dtype=torch.long)


def append_experience(experience):
    state, action, reward, next_state, done = experience
    target = reward
    if done == 0:
        target = target + target_net(next_state).max(1)[0].detach()
    error = (target - policy_net(state).gather(1, action)).detach()
    memory.add(error=error.squeeze().cpu(), sample=experience)


episode_durations = []
episode_rewards = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    # rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
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

    batch, idxs, is_weights = memory.sample(BATCH_SIZE)
    is_weights = torch.tensor(
        is_weights, device=device, dtype=torch.float32).to(device)
    batch = [*zip(*batch)]

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_states_batch = torch.cat(batch[3])
    dones_batch = torch.cat(batch[4])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # Compute the expected Q values
    next_state_values = target_net(next_states_batch).max(1)[0].detach()
    expected_state_action_values = (
        next_state_values * GAMMA) * (1 - dones_batch) + reward_batch
    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    # Compute Huber loss
    # criterion = nn.SmoothL1Loss()
    # loss = criterion(state_action_values, expected_state_action_values)
    loss = (is_weights * F.smooth_l1_loss(state_action_values,
            expected_state_action_values, reduction='none')).mean()

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    # update priorities
    error = torch.abs(expected_state_action_values -
                      state_action_values).detach().squeeze().cpu()
    for i in range(BATCH_SIZE):
        memory.update(idxs[i], error[i])


def motivation(state, action):
    # buckets state assumed - size 8,2
    global steps_done
    x = state[:, 0] * torch.exp(-0.1 * state[:, 1])
    motivation = 0.001 * (action != 0) * torch.max(x)
    return motivation


if not os.path.exists('targetnet_perdqn2'):
    os.makedirs('targetnet_perdqn2')  # save folder for value networks


def make_state(observation, info):
    observation[:, 0] /= 100
    observation[:, 1] /= 1000
    rel_cordinates = np.array(info['relative_coordinates'])/10000
    x = np.concatenate([observation.flatten(), rel_cordinates])
    return torch.tensor([x], device=device, dtype=torch.float32)


env.verbose = True
# env.MAX_STEPS = 400*60
num_episodes = 250
for i_episode in range(num_episodes):

    # Initialize the environment and state
    observation, done = env.reset()
    info = env.get_info()

    state = make_state(observation, info)
    for t in count():
        if t % 1000 == 0:
            print(
                f'episode = {i_episode}, count = {t}, reward = {env.cummulative_reward}')
        # Select and perform an action
        action = select_action(state, episode=i_episode)
        next_observation, env_reward, done, next_info = env.step(action.item())

        # motivation_reward = motivation(state, action)
        # reward = torch.tensor([env_reward+motivation_reward], device=device, dtype=torch.float32)
        reward = torch.tensor([env_reward], device=device, dtype=torch.float32)
        done = torch.tensor([done], device=device, dtype=torch.int)

        # Observe new state
        next_state = make_state(next_observation, next_info)

        # Store the transition in memory
        append_experience((state, action, reward, next_state, done))

        # Move to the next state
        state = next_state

        # env.render()
        if done:
            print("episode: ", i_episode, " reward: ", env.cummulative_reward)
            episode_durations.append(t + 1)
            episode_rewards.append(env_reward)
            break

    for i in range(100):
        # Perform one step of the optimization (on the policy network)
        optimize_model()

    with open(f'perdqn2_reward.txt', 'a') as f:
        f.write(str(env.cummulative_reward))
        f.write('\n')

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        torch.save(target_net.state_dict(),
                   f'targetnet_perdqn2/targetnet_ep{i_episode}.pth')

print('Complete')

env.close()
