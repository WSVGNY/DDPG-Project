from collections import deque
import numpy as np
import random

# Based on https://arxiv.org/abs/1509.02971
# As well as implementations from Patrick Emami, Hugo Germain &  Machine Learning with Phil

class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer = deque()
        self.buffer_size = buffer_size
        self.count = 0
    
    def store(self, state, action, reward, done, new_state):
        if self.count < self.buffer_size:
            self.count += 1
        else:
            self.buffer.popleft()
        
        self.buffer.append((state, action, reward, done, new_state))
    
    def sample(self, batch_size):
        experiences = random.sample(self.buffer, min(self.count, batch_size))
        
        states = np.array([_[0] for _ in experiences])
        actions = np.array([_[1] for _ in experiences])
        rewards = np.array([_[2] for _ in experiences])
        dones = np.array([_[3] for _ in experiences])
        new_states = np.array([_[4] for _ in experiences])

        return states, actions, rewards, dones, new_states
    
    def get_size(self):
        return self.count

    def clear(self):
        self.buffer = deque()
        self.count = 0
