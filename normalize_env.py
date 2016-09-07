import numpy as np
from gym.spaces import Box


class Normalization(object):
    def __init__(self, env):
        self.wrapped_env = env

        assert isinstance(env.action_space, Box)

        self.action_lb = self.wrapped_env.action_space.low
        self.action_ub = self.wrapped_env.action_space.high


        self.monitor = self.wrapped_env.monitor
        self.observation_space = self.wrapped_env.observation_space

        # redefine the action_space, so that all the action is with in the bound [-1,1]
        bounds = np.ones(self.wrapped_env.action_space.shape)
        self.action_space = Box(-1*bounds, bounds)


    def step(self, action):
        # the actual equation is 
        # scaled_action = self.action_lb + (action - (-1.0)) * (1.0 - (-1.0)) * (self.action_ub-self.action_lb)
        scaled_action = self.action_lb + (action + 1.0) * 0.5 * (self.action_ub-self.action_lb)

        # this probably is not necessary, since each environment should do the clipping before step
        scaled_action = np.clip(scaled_action, self.action_lb, self.action_ub)


        return self.wrapped_env.step(action)


    def reset(self):
        return self.wrapped_env.reset()

    def render(self):
        return self.wrapped_env.render()
