import berry_field
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
from constants import *

env = get_env.get_env()
bounding_boxes = env.bounding_boxes
berry_collision_tree = env.berry_collision_tree

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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

    def __init__(self, input_size, hidden_sizes, output_size):
        super(DQN, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_sizes[0])
        self.L2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.L3 = nn.Linear(hidden_sizes[1], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.L3(x)
        return x


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
input_size = 175
learning_rate = 0.001
hiddensizes = [100,50]
n_actions = 9


policy_net = DQN(input_size, hiddensizes, n_actions).to(device)
target_net = DQN(input_size, hiddensizes, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


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
    display.clear_output(wait=True)
    display.display(plt.gcf())


##Training Loop ------------

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
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()



num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()

    
    state = initial_state
    for t in count():
        # Select and perform an action
        action = select_action(state)
        observation, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # Observe new state
        #observation's new state
        next_state = 0 

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()


# class DQN():
#     def init(self, env, gamma, epsilon, tau, bufferSize, updateFrequency, MAX_TRAIN_EPISODES, MAX_EVAL_EPISODES, explorationStrategyTrainFn,
#     explorationStrategyEvalFn, optimizerFn):

#         self.env = env
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.tau = tau
#         self.bufferSize = bufferSize
#         self.updateFrequency = updateFrequency
#         self.MAX_TRAIN_EPISODES = MAX_TRAIN_EPISODES
#         self.MAX_EVAL_EPISODES = MAX_EVAL_EPISODES
#         self.explorationStrategyTrainFn = explorationStrategyTrainFn
#         self.explorationStrategyEvalFn = explorationStrategyEvalFn
#         self.optimizerFn = optimizerFn


#         self.nnTarget = ConvNet(S, A, hidDim, activationFunction)
#         self.nnOnline = ConvNet(S, A, hidDim, activationFunction)
#         copyNetworks(self.nnOnline, self.nnTarget)
#         self.initBookeeping()
#         self.rBuffer = ReplayBuffer(bufferSize)

#     def runDQN():
#         resultTrain = self.trainAgent()
#         resultsEval = self.evaluateAgent()
#         plotResults()
#         return result, final_eval_score, training_time, wallclock_time

#     def trainAgent(self):
#         copyNetworks(self.nnOnline, self.nnTarget)
#         for e in range(self.MAX_TRAIN_EPISODES):
#             s, done = self.env.reset()
#             self.rBuffer.collectExperiences(self.env, s, self.ExplorationStrategyTrainFn)
#             experiences = self.rBuffer.sample(batchSize)
#             self.trainQN(experiences)
#             self.performBookeeping(train = True)
#             self.evaluateAgent(qNetwork, self.MAX_EVAL_EPISODES)
#             self.performBookeeping(train=False)
#             if e%self.updateFrequency == 0:
#                 copyNetwork(self.nnOnline, self.nnTarget)
    
#     def trainQN(self, experiences):
#         ss, a, rs, sNexts, dones = self.rBuffer.splitExperiences(experiences)
#         max_a_qs = self.nnTarget(sNexts).detach().max()
#         tdTargets = rs + self.gamma * max_a_qs * (1 - dones)
#         qs = self.nnOnline(ss).gather(a)
#         tdErrors = tdTargets - qs
#         loss = mean(0.5*(tdErrors)**2)
#         optimizerFn.init()
#         loss.backward()
#         optimizerFn.step()

#     def EvaluateAgent(self, qNetwork, MAX_EVAL_EPISODES):
#         rewards = []
#         for e in range(MAX_EVAL_EPISODES):
#             rs = 0
#             s, done = self.env.reset()
#             for c in count():
#                 a = self.explorationStrategyEvalFn(nnOnline, s)
#                 s, r, done = self.env.step(a)
#                 rs += r
#                 if done:
#                     rewards.append(rs)
#                     break
#         self.performBookeeping(train=False)
#         return mean(rewards), std(rewards)

#     def initBookKeeping(self):
#         pass
    
#     def performBookeeping(train=True):
#         pass


# class ReplayMemory(object):

#     def __init__(self, bufferSize):
#         self.memory = deque([],maxlen=bufferSize)

#     def store(self, experience):
#         pass

#     def length(self):
#         pass

#     def collectExperiences(self, env, s, explorationStrategy):
#         pass

#     def push(self, *args):   #Extra
#         """Save a transition"""
#         self.memory.append(Transition(*args))

#     def sample(self, batchSize):
#         return np.random.sample(self.memory, batchSize)
    
#     def splitExperiences(self, experiences):
#         pass

#     def __len__(self):
#         return len(self.memory)

