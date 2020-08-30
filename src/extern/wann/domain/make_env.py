import numpy as np
import gym
from matplotlib.pyplot import imread


def make_env(env_name, seed=-1, render_mode=False):
  env = gym.make(env_name)

  if (seed >= 0):
    extern.wann.domain.seed(seed)

  return env
