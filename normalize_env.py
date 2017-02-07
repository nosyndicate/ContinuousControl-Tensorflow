import numpy as np
from gym.spaces import Box
from gym import Wrapper

class Normalization(Wrapper):
    def __init__(self, env):
        super(Normalization, self).__init__(env)

        assert isinstance(env.action_space, Box)

        self.action_lb = self.env.action_space.low
        self.action_ub = self.env.action_space.high


        self.observation_space = self.env.observation_space

        # redefine the action_space, so that all the action is with in the bound [-1,1]
        bounds = np.ones(self.env.action_space.shape)
        self.action_space = Box(-1*bounds, bounds)


    def step(self, action):
        # the actual equation is 
        # scaled_action = self.action_lb + (action - (-1.0)) * (1.0 - (-1.0)) * (self.action_ub-self.action_lb)
        scaled_action = self.action_lb + (action + 1.0) * 0.5 * (self.action_ub-self.action_lb)

        # this probably is not necessary, since each environment should do the clipping before step
        scaled_action = np.clip(scaled_action, self.action_lb, self.action_ub)


        return self.env.step(scaled_action)


    
