import random
from .ddpg import DDPGAgent
from .utilities import ReplayBuffer, PrioritizedReplayBuffer
import torch
import torch.nn.functional as F
import numpy as np


class MultiAgentConfig():
    def __init__(self):
        self.seed = 0
        self.buffer_size = int(1e5)
        self.batch_size = 512
        self.gamma = 0.99
        self.tau = 2e-1

        self.actor_hidden_sizes = [256, 128]
        self.lr_actor = 1e-4
        self.critic_hidden_sizes = [256, 128]
        self.lr_critic = 3e-4
        self.critic_weight_decay = 0.

        self.mu = 0.
        self.theta = 0.15
        self.sigma = 0.2

        self.update_every = 5

        self.prioritized_replay = True
        self.beta = 0.4
        self.beta_decay = 10000
        self.alpha = 0.6

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __repr__(self):
        return "Agent Config:\n\tbuffer size: {}\tbatch size: {}\n\tgamma: {}\ttau: {}\n\tactor lr: {}\tcritic lr: {}\n\tmu: {}\ttheta: {}\tsigma: {}".format(self.buffer_size, self.batch_size, self.gamma, self.tau, self.lr_actor, self.lr_critic, self.mu, self.theta, self.sigma);


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def transpose_to_tensor(input_list):
    make_tensor = lambda x: torch.tensor(x, dtype=torch.float)
    return list(map(make_tensor, zip(*input_list)))

class MADDPG:
    def __init__(self, state_size, action_size, config):
        # super(MADDPG, self).__init__()
        super().__init__()

        self.config = config
        self.seed = random.seed(config.seed)

        self.maddpg_agent = [DDPGAgent(state_size, action_size, self.config),
                             DDPGAgent(state_size, action_size, self.config)]

        self.iter = 0
        self.learn_iter = 0

        self.beta_function = lambda x: min(1.0, self.config.beta + x * (1.0 - self.config.beta) / self.config.beta_decay)

        if self.config.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(self.config.buffer_size, self.config.alpha, self.config.seed)
        else:
            self.memory = ReplayBuffer(self.config.buffer_size, self.config.seed)

    def get_actors(self):
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0):
        obs_all_agents = torch.tensor(obs_all_agents, dtype=torch.float).to(self.config.device)
        actions = [np.clip(agent.act(obs, noise).cpu().data.numpy(), -1, 1) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        target_actions = [ddpg_agent.target_act(obs, noise) for ddpg_agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions

    def update_targets(self):
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.config.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.config.tau)

    def reset(self):
        for ddpg_agent in self.maddpg_agent:
            ddpg_agent.reset()

    def step(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        self.iter += 1

        if(len(self.memory) >= self.config.batch_size) and self.iter % self.config.update_every == 0:
            beta = self.beta_function(self.learn_iter)
            for i in range(len(self.maddpg_agent)):
                samples = self.memory.sample(self.config.batch_size, beta)
                self.update_tuned(samples, i)
            self.learn_iter += 1
            self.update_targets()

    def save_checkpoints(self, file_head):
        for i in range(len(self.maddpg_agent)):
            file_name = file_head + 'agent{}_'.format(i)
            torch.save(self.maddpg_agent[i].actor.state_dict(), file_name + '_actor.pth')
            torch.save(self.maddpg_agent[i].critic.state_dict(), file_name + '_critic.pth')

    def load_checkpoints(self, file_head):
        for i in range(len(self.maddpg_agent)):
            file_name = file_head + 'agent{}_'.format(i)
            self.maddpg_agent[i].actor.load_state_dict(torch.load(file_name + '_actor.pth'))
            self.maddpg_agent[i].critic.load_state_dict(torch.load(file_name + '_critic.pth'))


    def _prep_samples(self, samples):
        convert = lambda x: torch.tensor(x, dtype=torch.float).to(self.config.device)

        state, action, reward, next_state, done, weights, idx = samples

        state = np.rollaxis(state, 1)
        next_state = np.rollaxis(next_state, 1)
        state_full = np.hstack(state)
        next_state_full = np.hstack(next_state)

        state = convert(state)
        state_full = convert(state_full)
        action = convert(action)
        reward = convert(reward)
        next_state = convert(next_state)
        next_state_full = convert(next_state_full)
        done = convert(np.float32(done))
        weights = convert(weights)

        return state, state_full, action, reward, next_state, next_state_full, done, idx, weights

    def update_tuned(self, samples, agent_number):

        state, state_full, action, reward, next_state, next_state_full, done, idx, weights = self._prep_samples(samples)

        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        next_actions = self.target_act(next_state)
        next_actions = torch.cat(next_actions, dim=1).detach()

        target_critic_input = torch.cat((next_state_full,next_actions), dim=1)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        q_targets = reward[..., agent_number].unsqueeze(1) + self.config.gamma * q_next * (1 - done[..., agent_number].unsqueeze(1))
        critic_input = torch.cat((state_full, action.view(self.config.batch_size, -1)), dim=1)
        q_expected = agent.critic(critic_input)

        if self.config.prioritized_replay:
            critic_loss = (q_targets.squeeze(1).detach() - q_expected.squeeze(1)).pow(2) * weights
            priors = critic_loss + 1e-5
            critic_loss = critic_loss.mean()
        else:
            critic_loss = F.mse_loss(q_expected, q_targets.detach())
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative

        online_actions = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(state) ]

        online_actions = torch.cat(online_actions, dim=1)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((state_full, online_actions), dim=1)

        actor_loss = -agent.critic(q_input2).mean()

        # get the policy gradient
        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(agent.actor.parameters(),0.5)
        agent.actor_optimizer.step()

        if self.config.prioritized_replay:
            self.memory.update_priorities(idx, priors.data.cpu().numpy())

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
