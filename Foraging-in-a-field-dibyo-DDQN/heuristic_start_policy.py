import numpy as np
import random
import torch
import math

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



EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

steps_done = 0
def select_action(state, policy_net):
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
        action = heuristicpolicy(state, distance_discount=0.8)
        return torch.tensor([[action]], device=device, dtype=torch.long)
