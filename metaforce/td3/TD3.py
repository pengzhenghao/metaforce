# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
# This file is mostly copied from: https://github.com/sfujim/TD3
import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        if isinstance(state_dim, tuple):
            raise NotImplementedError()
            # Image input!
            # self.actor = ConvActor(state_dim, action_dim, max_action).to(
            # device)
            # self.critic = ConvCritic(state_dim, action_dim).to(device)
            # self.obs_rgb = True
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)
            self.obs_rgb = False

        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4
        )

        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4
        )

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        if self.obs_rgb:
            state = torch.FloatTensor(state.reshape(1, *state.shape)).to(device)
        else:
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def select_action_in_batch(self, state):
        if self.obs_rgb:
            state = torch.FloatTensor(state.reshape(1, *state.shape)).to(device)
        else:
            state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        result = dict()

        # Sample replay buffer
        _, state, action, next_state, reward, not_done, _ = \
            replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (self.actor_target(next_state) +
                           noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)
        result["critic_q1"] = current_Q1.mean().item()
        result["target_q"] = target_Q.mean().item()
        result["target_q_max"] = torch.max(target_Q1, target_Q2).mean().item()
        result["target_q_diff"] = (
                torch.max(target_Q1, target_Q2) - torch.min(target_Q1,
                                                            target_Q2)
        ).mean().item()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        result["critic_loss"] = critic_loss.item()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            result["actor_loss"] = actor_loss.item()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        return result

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(
            self.critic_optimizer.state_dict(), filename + "_critic_optimizer"
        )
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(
            self.actor_optimizer.state_dict(), filename + "_actor_optimizer"
        )

    def load(self, filename):
        load_func = lambda path: torch.load(
            path,
            map_location=torch.device('cpu') \
                if not torch.cuda.is_available() else None
        )

        if os.path.exists(filename + "_critic"):
            pass
        else:
            # Assume it is a directory looks like "../models/"
            env_names = [
                path[path.find("Safexp"):].split("_")[0]
                for path in os.listdir(filename)
            ]
            assert len(set(env_names)) == 1
            suffix = [p for p in os.listdir(filename) if p.endswith("_actor")]
            assert len(suffix) == 1, os.listdir(filename)
            suffix = suffix[0]
            suffix = os.path.join(
                filename, suffix.split("_actor")[0]
            )
            filename = suffix

        print("We are restoring checkpoints from {}".format(filename))

        self.critic.load_state_dict(load_func(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            load_func(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(load_func(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            load_func(filename + "_actor_optimizer")
        )
        self.actor_target = copy.deepcopy(self.actor)
        return filename
