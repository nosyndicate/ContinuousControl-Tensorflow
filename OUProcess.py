import numpy as np
import numpy.random as nr


# Implemented based on rllab
# https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py



class OUProcess(object):
    def __init__(self, action_dim, sigma=0.2, mu=0, theta=0.15):

        self.mu = mu
        self.theta = theta
        self.sigma = sigma

        self.action_dim = action_dim

        self.state = np.ones(action_dim) * self.mu
        self.reset()

    def add_noise(self, action, lower, upper):
        x = self.state
        dw = nr.randn(len(x))
        dx = self.theta * (self.mu - x) + self.sigma * dw
        self.state = x + dx

        return np.clip(action + self.state, lower, upper)


    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

