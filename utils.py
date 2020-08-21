import time

import numpy as np
import torch


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_fn, seed, eval_episodes=10):
    # eval_env = gym.make(env_name)
    eval_env = env_fn()
    eval_env.seed(seed + 100)
    avg_cost = 0.
    avg_reward = 0.
    now = time.time()
    for _ in range(eval_episodes):
        episode_cost = 0
        state, done = eval_env.reset(), False

        last_state = np.zeros_like(state)
        last_action = np.zeros(eval_env.action_space.shape[0])

        while not done:

            if hasattr(policy, "context_mode") and policy.context_mode:
                if "transition" in policy.context_mode:
                    transition = np.concatenate(
                        [last_state, last_action, state])
                else:
                    transition = state
                action = policy.select_action(np.array(state),
                                              np.array(transition))
            else:
                action = policy.select_action(np.array(state))
            state, reward, done, info = eval_env.step(action)
            avg_reward += reward
            episode_cost += info.get('cost', 0.0)
        avg_cost += episode_cost
    avg_reward /= eval_episodes
    avg_cost /= eval_episodes
    print("---------------------------------------")
    print(
        f"Evaluation over {eval_episodes} episodes. Reward: {avg_reward:.3f}. "
        f"Cost: {avg_cost:.3f}. Time {time.time() - now:.3f}s."
    )
    print("---------------------------------------")
    return avg_reward, avg_cost


def eval_policy_ipd(policy, env_name, seed, eval_episodes=10, threshold=None):
    raise NotImplementedError("You should not use this function.")
    #     eval_env = gym.make(env_name)
    #     eval_env.seed(seed + 100)
    #     avg_reward = 0
    #     for _ in range(eval_episodes):
    #         episode_cost = 0
    #         state, done = eval_env.reset(), False
    #         while not done:
    #             action = policy.select_action(np.array(state))
    #             next_state, reward, done, info = eval_env.step(action)
    #             cost = info['cost']
    #             episode_cost += cost
    #             if threshold is not None:
    #                 done = done or episode_cost >= threshold
    #                 # TODO early stop is not processed here
    #             avg_reward += reward
    #     avg_reward /= eval_episodes
    #     print("---------------------------------------")
    #     print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    #     print("---------------------------------------")
    #     return avg_reward


def _auto_padding(data, index, seq_len):
    if index < 0:
        index = 0
    if index > len(data):
        index = len(data)
    if index >= seq_len:
        s = data[index - seq_len: index]
    else:
        s = np.ones([seq_len - index, data.shape[1]], dtype=np.float32)
        s = np.concatenate([s, data[0: index]], axis=0)
    return s


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6), use_rnn=False,
                 rnn_state_dim=None, rnn_seq_len=None, rnn_use_transition=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        if isinstance(state_dim, tuple):
            # RGB input
            self.state = np.zeros((max_size, *state_dim), dtype=np.uint8)
            self.next_state = np.zeros((max_size, *state_dim), dtype=np.uint8)
            self.obs_rgb = True
        else:
            self.state = np.zeros((max_size, state_dim), dtype=np.float32)
            self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
            self.obs_rgb = False
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)
        self.cost = np.zeros((max_size, 1), dtype=np.float32)
        self.rnn_use_transition = bool(rnn_use_transition)
        # self.quality_indicator = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.use_rnn = use_rnn
        self.rnn_state = None
        self.rnn_state_dim = rnn_state_dim
        self.rnn_seq_len = rnn_seq_len
        if use_rnn:
            assert rnn_state_dim is not None
            assert rnn_seq_len is not None
            # self.rnn_state = np.zeros((max_size, rnn_state_dim))

    def add(self, state, action, next_state, reward, done, cost,
            # rnn_state=None
            ):
        self.state[self.ptr] = state
        self.action[self.ptr] = action

        if self.obs_rgb:
            self.state[self.ptr] = (state * 255).astype(np.uint8)
            self.next_state[self.ptr] = (next_state * 255).astype(np.uint8)
        else:
            self.state[self.ptr] = state
            self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.cost[self.ptr] = cost
        # self.quality_indicator[self.ptr] = quality

        # if self.use_rnn:
        #     assert rnn_state is not None
        #     self.rnn_state[self.ptr] = rnn_state

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        if self.obs_rgb:
            state = (self.state[ind] / 255).astype(np.float32)
            next_state = (self.next_state[ind] / 255).astype(np.float32)
        else:
            state = self.state[ind]
            next_state = self.next_state[ind]

        ret = (
            torch.FloatTensor(state).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(next_state).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.cost[ind]).to(self.device)
            # torch.FloatTensor(self.quality_indicator[ind]).to(self.device)
        )

        if self.use_rnn:

            # TODO: it is possible that at the begining of a new episode
            #  you will use the hidden state from a unknown previous episode.
            # Prepared a sequence of hidden state
            rnn_data = []
            for i in ind:
                # data for RNN
                state = _auto_padding(self.state, i, self.rnn_seq_len + 1)
                not_done = _auto_padding(self.not_done, i, self.rnn_seq_len + 1)

                if not_done.min() < 0.01:  # Some done = True in this sequence
                    # There still has one case that we ignore: an trajectory
                    # with multiple done.
                    state[:np.argmin(not_done) + 1] = 0.0

                if self.rnn_use_transition:
                    last_action = _auto_padding(
                        self.action, i - 1, self.rnn_seq_len + 1)
                    last_state = _auto_padding(
                        self.state, i - 1, self.rnn_seq_len + 1)
                    if not_done.min() < 0.01:
                        last_action[:np.argmin(not_done) + 1 + 1] = 0.0
                        last_state[:np.argmin(not_done) + 1 + 1] = 0.0
                    state = np.concatenate(
                        [last_state, last_action, state], axis=1)

                rnn_data.append(state)

            rnn_data = torch.FloatTensor(np.stack(rnn_data, axis=1)).to(
                self.device)

            # next_rnn_data = []
            # for i in ind:
            #     # next_state = _auto_padding(self.next_state, i,
            #     self.rnn_seq_len)
            #     # not_done = _auto_padding(self.not_done, i, self.rnn_seq_len)
            #     # if not_done.min == 0.0:
            #         # There still has one case that we ignore: an trajectory
            #         # with multiple done.
            #         # state[:np.argmin(not_done)] = 0.0
            #
            #     if self.rnn_use_transition:
            #         next_last_action = _auto_padding(self.action, i,
            #         self.rnn_seq_len)
            #         next_last_state = _auto_padding(self.state, i,
            #         self.rnn_seq_len)
            #         if not_done.min == 0.0:
            #             next_last_action[:np.argmin(not_done) + 1] = 0
            #             next_last_state[:np.argmin(not_done) + 1] = 0
            #         next_state = np.concatenate(
            #             [next_last_state, next_last_action, next_state],
            #             axis=1)
            #
            #     next_rnn_data.append(next_state)
            # next_rnn_data = torch.FloatTensor(
            #     np.stack(next_rnn_data, axis=1)).to(self.device)

            # ret += (rnn_data, next_rnn_data)

            ret += (rnn_data,)

        return ret

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.size])
        np.save(f"{save_folder}_action.npy", self.action[:self.size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}_cost.npy", self.cost[:self.size])
        # np.save(f"{save_folder}_quality_indicator.npy",
        # self.quality_indicator[:self.size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

        if self.use_rnn:
            np.save(f"{save_folder}_rnn_state.npy", self.rnn_state[:self.size])

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")
        cost_buffer = np.load(f"{save_folder}_cost.npy")
        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        self.state[:self.size] = np.load(f"{save_folder}_state.npy"
                                         )[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}_action.npy"
                                          )[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}_next_state.npy"
                                              )[:self.size]
        self.reward[:self.size] = reward_buffer[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}_not_done.npy"
                                            )[:self.size]
        self.cost[:self.size] = cost_buffer[:self.size]
        # self.quality_indicator[:self.size] = np.load(f"{
        # save_folder}_quality_indicator.npy")[:self.size]

        if self.use_rnn:
            self.rnn_state[:self.size] = np.load(
                f"{save_folder}_rnn_state.npy")[:self.size]
