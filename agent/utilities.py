import random
import copy
from collections import namedtuple, deque

import torch
import numpy as np

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, scale=0.1, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.size = size
        self.scale = scale
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()


class ReplayBuffer:
    def __init__(self, size, seed=0):
        self.buffer = deque(maxlen=size)
        self.seed = random.seed(seed)
        self.size = size

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, beta):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))

        return np.concatenate(state), action, reward, np.concatenate(next_state), done, np.ones(self.size), None

    def update_priorities(self, indices, updated_priorities):
        return

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, seed=0):
        self.seed = random.seed(seed)
        self.alpha = alpha
        self.buffer = []
        self.size = size
        self.idx = 0
        self.priorities = np.zeros(self.size)

    def push(self, state, action, reward, next_state, done):
        # store in circular buffer like above
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        experience = (state, action, reward, next_state, done)

        if len(self.buffer) == 0:
            max_p = 1.0
        else:
            max_p = np.max(self.priorities)

        self.priorities[self.idx] = max_p

        if self.idx >= len(self.buffer):
            self.buffer.append(experience)
        else:
            self.buffer[self.idx] = experience

        self.idx = (self.idx + 1) % self.size

    def sample(self, batch_size, beta):
        priorities = self.priorities
        if len(self.buffer) < self.size:
            priorities = self.priorities[:self.idx]

        prob = np.power(priorities, self.alpha)
        prob /= np.sum(prob)

        index_list = np.random.choice(len(self.buffer), batch_size, p=prob)
        state, action, reward, next_state, done = zip(*[self.buffer[i] for i in index_list])

        is_weights = np.power((prob[index_list] * len(self.buffer)), -beta)
        is_weights /= np.max(is_weights)
        is_weights = np.array(is_weights, dtype=np.float32)

        return np.concatenate(state), action, reward, np.concatenate(next_state), done, is_weights, index_list

    def update_priorities(self, indices, updated_priorities):
        for idx, priority in zip(indices, updated_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)
