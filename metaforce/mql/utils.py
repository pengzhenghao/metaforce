import numpy as np
import torch

from metaforce.td3 import ReplayBuffer


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


class LearnableContextReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6),
                 rnn_state_dim=None, rnn_seq_len=None, rnn_use_transition=None):
        super(LearnableContextReplayBuffer, self).__init__(
            state_dim, action_dim, max_size
        )
        self.rnn_use_transition = bool(rnn_use_transition)
        self.rnn_state = None
        self.rnn_state_dim = rnn_state_dim
        self.rnn_seq_len = rnn_seq_len
        assert rnn_state_dim is not None
        assert rnn_seq_len is not None
        self.rnn_state = np.zeros((max_size, rnn_state_dim))

    def add(self, state, action, next_state, reward, done, cost,
            rnn_state=None):
        assert rnn_state is not None
        self.rnn_state[self.ptr] = rnn_state
        super(LearnableContextReplayBuffer, self).add(
            state, action, next_state, reward, done, cost
        )

    def sample(self, batch_size):
        ret = super(LearnableContextReplayBuffer, self).sample(batch_size)
        ind = ret[0]

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

        ret += (rnn_data,)
        return ret

    def save(self, save_folder):
        super(LearnableContextReplayBuffer, self).save(save_folder)
        np.save(f"{save_folder}_rnn_state.npy", self.rnn_state[:self.size])

    def load(self, save_folder, size=-1):
        super(LearnableContextReplayBuffer, self).load(save_folder, size)
        self.rnn_state[:self.size] = np.load(
            f"{save_folder}_rnn_state.npy")[:self.size]


class RandomContextReplayBuffer(ReplayBuffer):
    def __init__(self, state_dim, action_dim, max_size=int(1e6),
                 context_dim=None):
        super(RandomContextReplayBuffer, self).__init__(
            state_dim, action_dim, max_size
        )
        assert context_dim is not None
        self.context = np.zeros((max_size, context_dim), dtype=np.float32)

    def add(self, state, action, next_state, reward, done, cost, context=None):
        # We should add things before super, because the self.ptr is changed
        # during original update.
        assert context is not None
        self.context[self.ptr] = context

        super(RandomContextReplayBuffer, self).add(
            state, action, next_state, reward, done, cost
        )

    def sample(self, batch_size):
        ret = super(RandomContextReplayBuffer, self).sample(batch_size)
        ind = ret[0]
        ret += (
            torch.FloatTensor(self.context[ind]).to(self.device),
            torch.FloatTensor(
                self.context[np.maximum(ind + 1, len(self.context) - 1)]
            ).to(self.device)
        )
        return ret

    def save(self, save_folder):
        super(RandomContextReplayBuffer, self).save(save_folder)
        np.save(f"{save_folder}_context.npy", self.context[:self.size])

    def load(self, save_folder, size=-1):
        super(RandomContextReplayBuffer, self).load(save_folder, size)
        self.context[:self.size] = np.load(
            f"{save_folder}_context.npy")[:self.size]
