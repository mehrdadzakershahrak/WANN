from collections import deque
import random


class ReflectMem(object):
    def __init__(self, cap):
        self._store = deque(maxlen=cap)

    def add(self, obs):
        self._store.append(obs)

    def sample(self, batch_size):
        return random.sample(self._store, batch_size)

    def clear(self):
        self._store.clear()

    def size(self):
        return len(self._store)
