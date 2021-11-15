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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type, device)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Linear(input_size,35)
        self.conv2 = nn.Linear(35, 16)
        self.conv3 = nn.Linear(16, output_size)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(device)
        #print(x.shape)
        #print(x.shape)
        # #x = torch.permute(x,(0,2,1))
        # # x = x.unsqueeze(0)
        # # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        # x = x.squeeze()
        # print(x.shape)
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.1
EPS_END = 0.01
EPS_DECAY = 200
TARGET_UPDATE = 10
action_space_size=9
input_size = 35*5

# Get number of actions from gym action space
n_actions = action_space_size

policy_net = DQN(input_size, n_actions).to(dtype=float).to(device)
target_net = DQN(input_size, n_actions).to(dtype=float).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

def load(i):
    policy_net.load_state_dict(torch.load(f'policynet/policynet_ep{i}.pth'))
    target_net.load_state_dict(torch.load(f'targetnet/targetnet_ep{i}.pth'))

# load(1)

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

def getTrueAngles(directions, referenceVector=[0,1]):
    curls = np.cross(directions, referenceVector)
    dot = np.dot(directions, referenceVector)
    angles = np.arccos(dot)*180/np.pi
    args0 = np.argwhere(np.bitwise_not((curls > 0)|(curls == 0)&(dot==1)))
    angles[args0] = 360-angles[args0]
    return angles


sectors = [((x-22.5)%360, x+22.5) for x in range(0,360,45)]
lastaction = 1
def heuristicpolicy(obs, distance_discount=0.8):
    """obs: [[isBerry, direction(2 cols), distance, size]]"""
    global lastaction
    berries = np.argwhere(np.isclose(obs[:,0], 1))[:,0]
    if berries.shape[0]==0: return lastaction

    obs = obs[berries]
    directions, distances, sizes = obs[:,1:3], obs[:,3], obs[:,4]
    angles = getTrueAngles(directions, [0,1])
    juices = np.zeros(8)
    for i, sector in enumerate(sectors):
        if sector[0] < sector[1]:
            args = np.argwhere((angles>=sector[0])&(angles<=sector[1]))
        else:
            args = np.argwhere((angles>=sector[0])|(angles<=sector[1]))
        args = np.squeeze(args)
        juicediscount = np.power(distance_discount, distances[args])
        discounted_juice = np.dot(sizes[args], juicediscount)
        juices[i] = discounted_juice

    action = np.argmax(juices)+1
    lastaction = action
    return action



def select_action(state, episode_number):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * episode_number / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            #print(policy_net(state).max().item())
            #print(policy_net(state))
            #print(state.shape)
            return policy_net(state).argmax(), 1
            #return policy_net(state).max(1)[1].view(1, 1)
    else:
        # action = heuristicpolicy(state, distance_discount=0.8)
        # return torch.tensor([[action]], device=device)
        return torch.tensor(np.random.randint(9,size=1)[0],device = device), 25


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
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.stack(batch.state)
    action_batch = torch.stack(batch.action)
    reward_batch = torch.stack(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #print(action_batch.shape)
    action_batch = action_batch.type(torch.int64)
    #print(policy_net(state_batch).shape,action_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch.reshape(256,1))

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=float)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze(1)
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 1000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    
    state, done = env.reset()
    state = state.flatten()
    state = torch.from_numpy(state).to(device)
    #state = state.double()
    k = 0
    action = 0
    for t in count():
        # Select and perform an action
        
        #print("Episode {} step {}".format(i_episode,t))
        if i_episode<=-1:
            action = torch.tensor(heuristicpolicy(env.unordered_observation()),device=device)
        else:
            if(not k):
                action, k = select_action(state, i_episode)
            k = k - 1  
        #print(action)
        next_state, reward, done, _ = env.step(action)
        bestBerry = np.min(next_state[:,3])
        reward += np.exp(-0.01*bestBerry)
       # env.render()
        next_state = next_state.flatten()
        next_state = torch.from_numpy(next_state).to(device)
        #next_state = next_state.double()
        reward = torch.tensor([reward], device=device)

        #print(reward)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if len(memory)>=BATCH_SIZE:
            optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    print("Episode {} done, cumulative reward {}".format(i_episode+1,env.cummulative_reward))

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    with open(f'policynet/policynet_ep{i_episode}.pth', 'wb') as f:
        torch.save(policy_net.state_dict(), f)
    with open(f'targetnet/targetnet_ep{i_episode}.pth', 'wb') as f:
        torch.save(target_net.state_dict(), f)
print('Complete')
env.render()
plt.ioff()
plt.show()