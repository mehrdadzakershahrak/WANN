from rlkit.data_management.replay_buffer import ReplayBuffer
import collections
import random


class Mem(ReplayBuffer):
    def __init__(self, size):
        self._buffer = collections.deque(size)

    def add_sample(self, obs, act, reward, next_obs, done):
        # TODO: add priority mechanism
        self._buffer.appendleft((obs, act, reward, next_obs, done))

    def random_batch(self, batch_size):
        return random.sample(self._buffer, batch_size)
