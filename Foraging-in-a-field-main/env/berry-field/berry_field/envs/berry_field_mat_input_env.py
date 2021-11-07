from functools import cmp_to_key

import gym
import numpy as np
import copy
import pyglet
from gym.envs.classic_control import rendering

from .utils.collision_tree import collision_tree
from .utils.renderingViewer import renderingViewer

MAX_DISPLAY_SIZE = (16*80, 9*80) # (width, height)
OBSHAPE = 35

class BerryFieldEnv_MatInput(gym.Env):
    def __init__(self,
                 file_paths,
                 num_berries, num_patches,
                 field_size, patch_size, agent_size, observation_space_size,
                 drain_rate, reward_rate,
                 max_steps,
                 initial_state, circular_berries=True, circular_agent=True,
                 observation_type = "segmented"):

        super(BerryFieldEnv_MatInput, self).__init__()

        # Checking for the correct format of the file_path argument
        if len(file_paths) != 2:
            raise Exception("file_paths should be a list of length 2")

        # Initializing variables
        self.FIELD_SIZE = field_size
        self.AGENT_SIZE = agent_size
        self.INITIAL_STATE = initial_state
        self.DRAIN_RATE = drain_rate
        self.REWARD_RATE = reward_rate
        self.MAX_STEPS = max_steps
        self.OBSERVATION_SPACE_SIZE = observation_space_size
        self.CIRCULAR_BERRIES = circular_berries
        self.CIRCULAR_AGENT = circular_agent
        self.OBSERVATION_TYPE = observation_type

        self.done = False
        self.state = initial_state
        self.num_steps = 0
        self.action_space = gym.spaces.Discrete(9)

        self.viewer = None

        self.cummulative_reward = 0.5
        self.observation = None
        self.lastaction = 0

        self.action_switcher = {
            0: (0, 0),
            1: (0, 1),
            2: (1, 1),
            3: (1, 0),
            4: (1, -1),
            5: (0, -1),
            6: (-1, -1),
            7: (-1, 0),
            8: (-1, 1)
        }

        berry_data = self.read_csv(file_paths)
        bounding_boxes, boxIds = self.create_bounding_boxes_and_Ids(berry_data)
        self.berry_radii = berry_data[:,0]/2 # [x,y,width,height]
        self.BERRY_COLLISION_TREE = collision_tree(bounding_boxes, boxIds, self.CIRCULAR_BERRIES, self.berry_radii)
        self.berry_collision_tree = copy.deepcopy(self.BERRY_COLLISION_TREE)


    def reset(self):
        if self.viewer: self.viewer.close()
        self.done = False
        self.state = self.INITIAL_STATE
        self.num_steps = 0
        self.viewer = None
        self.cummulative_reward = 0.5
        self.observation = self.unordered_observation()
        self.lastaction = 0
        self.berry_collision_tree = copy.deepcopy(self.BERRY_COLLISION_TREE)
        return self.observation,self.done


    def step(self, action):
        action = action.item()
        self.num_steps+=1
        self.lastaction = action
        movement = self.action_switcher[action]
        x = self.state[0] + movement[0]
        y = self.state[1] + movement[1]
        self.state = (  min(max(0, x), self.FIELD_SIZE[0]), 
                        min(max(0, y), self.FIELD_SIZE[1]) )
        reward = self.pick_collided_berries()  - self.DRAIN_RATE*(action != 0)
        observation = self.unordered_observation()
        # observation = self.ordered_observation()
        self.cummulative_reward += reward
        self.observation = observation
        self.done = True if self.num_steps >= self.MAX_STEPS else False
        if self.done and self.viewer is not None: self.viewer = self.viewer.close()
        return observation, reward, self.done, {}


    def segmented_observation(self):
        pass


    def ordered_observation(self):
        """ unoredered_observation sorted clockwise """
        observation = np.zeros((OBSHAPE, 5))
        boxIds, boxes = self.get_Ids_and_boxes_in_view((*self.state, *self.OBSERVATION_SPACE_SIZE))
        if len(boxIds) == 0: return observation

        agent_pos = np.array(self.state)
        directions = boxes[:,:2] - agent_pos
        distances = np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
        directions = directions/distances
        data = np.column_stack([np.ones_like(distances), directions, distances, boxes[:,-1]])
        args = self.argsort_clockwise(directions)
        observation[:data.shape[0],:] = data[args]
        return observation


    def unordered_observation(self):
        """ all visible berries are collated as colstack[isBerry, direction, distance, size]
            in the order they had been detected
            returns np array of shape (OBSHAPE,5) """
        agent_bbox = (*self.state, self.AGENT_SIZE, self.AGENT_SIZE)
        observation = np.zeros((OBSHAPE, 5))
        boxIds, boxes = self.get_Ids_and_boxes_in_view((*self.state, *self.OBSERVATION_SPACE_SIZE))
        if len(boxIds) == 0: return observation
        
        agent_pos = np.array(agent_bbox[:2])
        directions = boxes[:,:2] - agent_pos
        distances = np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
        directions = directions/distances
        data = np.column_stack([np.ones_like(distances), directions, distances, boxes[:,-1]])
        observation[:data.shape[0],:] = data
        return observation

    
    def pick_collided_berries(self):
        agent_bbox = (*self.state, self.AGENT_SIZE, self.AGENT_SIZE)
        boxIds, boxes = self.berry_collision_tree.find_collisions(agent_bbox, 
                                            self.CIRCULAR_AGENT, self.AGENT_SIZE/2, return_boxes=True)
        sizes = boxes[:,2] # boxes are an array with rows as [x,y, size, size]
        reward = self.REWARD_RATE * np.sum(sizes)
        self.berry_collision_tree.delete_boxes(list(boxIds))
        return reward


    def get_Ids_and_boxes_in_view(self, bounding_box):
        boxIds, boxes = self.berry_collision_tree.boxes_within_bound(bounding_box, return_boxes=True)
        return list(boxIds), boxes


    def create_bounding_boxes_and_Ids(self,berry_data):
        """ bounding boxes from berry-coordinates and size """
        bounding_boxes = np.column_stack([
            berry_data[:,1:], berry_data[:,0], berry_data[:,0]
        ])
        boxIds = np.arange(bounding_boxes.shape[0])
        return bounding_boxes, boxIds


    def argsort_clockwise(self, directions):
        tmepdirections = np.column_stack([np.arange(directions.shape[0]), directions])
        cmp = lambda x,y: self.isClockwise(x[1:], y[1:])
        args = np.array(sorted(tmepdirections, key=cmp_to_key(cmp)))[:,0]
        return args.astype(int)


    def isClockwisehelper(self, v):
        # partitions circle into two sub spaces
        rx,ry = (0,1) #reference vector
        x,y = v
        curl = rx*y - ry*x
        dot = x*rx + y*ry
        if curl < 0: return True
        if curl == 0 and dot == 1: return True
        return False


    def isClockwise(self,v1, v2):
        x1,y1,x2,y2=(*v1,*v2)
        curl = x1*y2 - x2*y1
        # dot = x1*x2 + y1*y2
        v1_in_A = self.isClockwisehelper(v1)
        v2_in_A = self.isClockwisehelper(v2)
        if(v1_in_A == v2_in_A):
            if(curl < 0): return -1
            return 1
        elif(v1_in_A and not v2_in_A):
            return -1
        return 1


    def read_csv(self, file_paths):
        # Constructing numpy arrays to store the coordinates of berries and patches
        berry_data = np.loadtxt(file_paths[0], delimiter=',') #[patch#, size, x,y]
        return berry_data[:, 1:]
    

    def render(self, mode="human", returnRGB=False):

        if self.done: 
            if self.viewer is not None: self.viewer = self.viewer.close()
            else: self.viewer = None
            return
        
        # berries in view
        screenw, screenh = self.OBSERVATION_SPACE_SIZE
        bounding_box = (*self.state, screenw, screenh)
        agent_bbox = (screenw/2, screenh/2, self.AGENT_SIZE, self.AGENT_SIZE)
        boxIds, boxes = self.get_Ids_and_boxes_in_view(bounding_box)
        boxes[:,0] -= self.state[0]-screenw/2; boxes[:,1] -= self.state[1]-screenh/2 
            
        # adjust for my screen size
        scale = min(1, min(MAX_DISPLAY_SIZE[0]/screenw, MAX_DISPLAY_SIZE[1]/screenh))
        screenw, screenh = int(screenw*scale), int(screenh*scale)
        if self.viewer is None: self.viewer = renderingViewer(screenw, screenh)
        self.viewer.transform.scale = (scale, scale)

        # draw berries
        if self.CIRCULAR_BERRIES:
            for center, radius in zip(boxes[:,:2], self.berry_radii[boxIds]):
                circle = rendering.make_circle(radius)
                circletrans = rendering.Transform(translation=center)
                circle.set_color(255,0,0)
                circle.add_attr(circletrans)
                self.viewer.add_onetime(circle)
        else:
            for x,y,width,height in boxes:
                l,r,b,t = -width/2, width/2, -height/2, height/2
                vertices = ((l,b), (l,t), (r,t), (r,b)) 
                box = rendering.FilledPolygon(vertices)
                boxtrans = rendering.Transform(translation=(x,y))
                box.set_color(255,0,0)
                box.add_attr(boxtrans)
                self.viewer.add_onetime(box)
        
        # draw agent
        if self.CIRCULAR_AGENT:
            agent = rendering.make_circle(self.AGENT_SIZE/2)
        else:
            p = self.AGENT_SIZE/2
            agentvertices =((-p,-p),(-p,p),(p,p),(p,-p))
            agent = rendering.FilledPolygon(agentvertices)
        agenttrans = rendering.Transform(translation=agent_bbox[:2])
        agent.set_color(0,0,0)
        agent.add_attr(agenttrans)
        self.viewer.add_onetime(agent)

        # draw boundary wall 
        l = bounding_box[0] - bounding_box[2]/2
        r = bounding_box[0] + bounding_box[2]/2 - self.FIELD_SIZE[0]
        b = bounding_box[1] - bounding_box[3]/2
        t = bounding_box[1] + bounding_box[3]/2 - self.FIELD_SIZE[1]
        top = bounding_box[3] - t
        right = bounding_box[2] - r
        if l<=0:
            line = rendering.Line(start=(-l, max(0, -b)), end=(-l,min(bounding_box[2], top)))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)
        if r>=0:
            line = rendering.Line(start=(right, max(0, -b)), end=(right,min(bounding_box[2], top)))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)
        if b<=0:
            line = rendering.Line(start=(max(0,-l), -b), end=(min(bounding_box[2], right),-b))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)
        if t>=0:
            line = rendering.Line(start=(max(0,-l), top), end=(min(bounding_box[1], right),top))
            line.set_color(0,0,255)
            self.viewer.add_onetime(line)

        # draw position and total reward
        label = pyglet.text.Label(f'x:{self.state[0]} y:{self.state[1]} a:{self.lastaction} \t total-reward:{self.cummulative_reward:.4f}', 
                                    x=screenw*0.1, y=screenh*0.9, color=(0, 0, 0, 255))
        self.viewer.add_onetimeText(label)

        return self.viewer.render(returnRGB)
