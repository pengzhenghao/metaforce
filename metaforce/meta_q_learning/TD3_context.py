import torch
import torch.nn as nn
import torch.nn.functional as F

from TD3 import TD3, device


class ContextModel(nn.Module):
    # The state management is on outside
    def __init__(self, state_dim, hidden_size=32, max_seq_len=20):
        super(ContextModel, self).__init__()
        self.hidden_size = hidden_size
        self.max_seq_len = 20
        self.rnn = nn.GRU(
            input_size=state_dim, hidden_size=self.hidden_size, batch_first=True
        )

    def forward(self, obs, state=None):
        obs = obs.to(device)
        if state is not None:
            state = state.to(device)
        if obs.dim() == 1:
            obs.unsqueeze(0)
        if obs.dim() == 2:
            obs.unsqueeze(0)

        if state is not None and state.dim() == 1:
            state.unsqueeze(0)
        if state is not None and state.dim() == 2:
            state.unsqueeze(0)

        return self.rnn(obs, state)


class TD3Context(TD3):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
            hidden_state_dim=24,
            context_mode=None
    ):
        assert context_mode is not None
        assert context_mode in [
            "add_both", "add_actor", "add_critic", "add_critic_transition",
            "add_actor_transition", "add_both_transition", "random", "disable"
        ], context_mode

        param_add_critic = ("both" in context_mode) or (
                "critic" in context_mode)
        param_add_actor = ("both" in context_mode) or ("actor" in context_mode)

        if ("transition" in context_mode) or (context_mode == "random"):
            context_state_dim = state_dim * 2 + action_dim
        else:
            context_state_dim = state_dim
        self.context_mode = context_mode

        self.context_model = ContextModel(
            context_state_dim, hidden_size=hidden_state_dim).to(device)

        state_dim = state_dim + hidden_state_dim
        self.hidden_state_dim = hidden_state_dim

        super().__init__(
            state_dim, action_dim, max_action, discount, tau, policy_noise,
            noise_clip, policy_freq
        )

        # self.context_model_optimizer = torch.optim.Adam(
        #     self.context_model.parameters(), lr=3e-4
        # )

        # Add the parameter of context model into the optimizer.
        actor_param = list(self.actor.parameters())
        if param_add_actor:
            actor_param += list(self.context_model.parameters())
        self.actor_optimizer = torch.optim.Adam(actor_param, lr=3e-4)

        critic_param = list(self.critic.parameters())
        if param_add_critic:
            critic_param += list(self.context_model.parameters())
        self.critic_optimizer = torch.optim.Adam(critic_param, lr=3e-4)

        self.prev_state = torch.zeros(
            [1, 1, self.hidden_state_dim], device=device, dtype=torch.float32
        )

    def select_action(self, state, transition):
        state = torch.FloatTensor(state.reshape(1, 1, -1)).to(device)
        transition = torch.FloatTensor(transition.reshape(1, 1, -1)).to(device)
        context, next_state = self.context_model(transition, self.prev_state)

        assert context.ndim == state.ndim == 3
        feed_state = torch.cat([state, context], dim=2)
        self.prev_state = next_state

        return self.actor(feed_state).cpu().data.numpy().flatten(), \
               context.detach().cpu().numpy()
        # return self.actor(state).cpu().data.numpy().flatten()

    def select_action_in_batch(self, state):
        raise NotImplementedError()
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done, cost, context, \
        next_context = replay_buffer.sample(batch_size)

        # current_context_all, _ = self.context_model(rnn_data)
        # current_context = current_context_all[-2]
        # next_context = current_context_all[-1]
        current_input = torch.cat([state, context], dim=1)

        # current_input = torch.cat([state, current_context], dim=1)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)

            next_input = torch.cat([next_state, next_context], dim=1)

            next_action = (self.actor_target(next_input) +
                           noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_input, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Build current input
        # current_context, _ = self.context_model(rnn_data)
        # current_context = current_context[-1]

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(current_input, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # actor_current_context, _ = self.context_model(rnn_data)

            # actor_current_context = actor_current_context[-1]
            # actor_current_context = actor_current_context[-2]

            actor_current_input = torch.cat([state, context], dim=1)

            # Compute actor loss
            actor_loss = -self.critic.Q1(
                actor_current_input, self.actor(actor_current_input)
            ).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
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

    def save(self, filename):
        super(TD3Context, self).save(filename)
        torch.save(
            self.context_model.state_dict(), filename + "_context_model"
        )

    def load(self, filename):
        filename = super(TD3Context, self).load(filename)

        load_func = lambda path: torch.load(
            path,
            map_location=torch.device('cpu') \
                if not torch.cuda.is_available() else None
        )

        self.context_model.load_state_dict(
            load_func(filename + "_context_model"))
