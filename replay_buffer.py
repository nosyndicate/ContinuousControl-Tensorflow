
from collections import deque
import random
import numpy as np

class ReplayBuffer(object):

    def __init__(self, size, state_dim, action_dim):
        
        self.replay_buffer_size = size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.states = np.zeros((size, state_dim))
        self.actions = np.zeros((size, action_dim))
        self.sprimes = np.zeros((size, state_dim))
        self.rewards = np.zeros((size, 1))
        # terminal are just one or zero
        self.terminals = np.zeros((size, 1), dtype='uint8')

        self.index = 0
        self.size = 0

    def add(self, s, a, sprime, r, t):

        self.states[self.index] = s
        self.actions[self.index] = a
        self.sprimes[self.index] = sprime
        self.rewards[self.index] = r
        self.terminals[self.index] = t


        self.index = (self.index + 1)%self.replay_buffer_size

        if self.size < self.replay_buffer_size:
            self.size += 1


    def get_size(self):
        return self.size

    def sample_batch(self, batch_size):
        # make sure we have enough samples
        assert self.size > batch_size

        # grab the indices from the available positions
        indices = np.random.randint(0, self.size, size=batch_size)

        return self.states[indices], self.actions[indices], self.sprimes[indices], self.rewards[indices], self.terminals[indices]



    def reset(self):
        self.index = 0
        self.size = 0;



