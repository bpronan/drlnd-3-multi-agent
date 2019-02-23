from .model import Network
from torch.optim import Adam

import torch
import numpy as np

from .utilities import OUNoise

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPGAgent:
    def __init__(self, state_size, action_size, config):
        super().__init__()
        # super(DDPGAgent, self).__init__()

        self.config = config

        self.actor = Network(state_size, config.actor_hidden_sizes[0],
                             config.actor_hidden_sizes[1], action_size, config.seed, actor=True).to(self.config.device)
        self.critic = Network(2 * (state_size + action_size), config.critic_hidden_sizes[0],
                              config.critic_hidden_sizes[1], 1, config.seed).to(self.config.device)

        self.target_actor = Network(state_size, config.actor_hidden_sizes[0],
                             config.actor_hidden_sizes[1], action_size, config.seed, actor=True).to(self.config.device)
        self.target_critic = Network(2 * (state_size + action_size), config.critic_hidden_sizes[0],
                              config.critic_hidden_sizes[1], 1, config.seed).to(self.config.device)


        self.noise = OUNoise(action_size, seed=self.config.seed, scale=1.0,
                             mu=self.config.mu, theta=self.config.theta, sigma=self.config.sigma)

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.config.lr_critic,
                                     weight_decay=self.config.critic_weight_decay)

    def act(self, obs, noise=0.0):
        action = self.actor(obs) + noise * self.noise.sample().to(self.config.device)
        return action

    def target_act(self, obs, noise=0.0):
        action = self.target_actor(obs) + noise * self.noise.sample().to(self.config.device)
        return action

    def reset(self):
        self.noise.reset()
