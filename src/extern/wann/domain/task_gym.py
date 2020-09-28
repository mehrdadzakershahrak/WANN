import random
import numpy as np
import sys
from extern.wann.domain.make_env import make_env
from extern.wann.neat_src import *
from stable_baselines3 import PPO, DDPG
from task import task
import torch as th


class GymTask():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1):
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness

    if agent_params is None:
    """
    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]

    # TODO: clean this up
    # HACK

    # Environment
    self.nReps = nReps
    self.maxEpisodeLength = game.max_episode_length
    self.actSelect = game.actionSelect
    if not paramOnly:
      self.env = make_env(game.env_name)
    
    # Special needs...
    self.needsClosed = (game.env_name.startswith("CartPoleSwingUp"))    
  
  def getFitness(self, wVec, aVec, view=False, nRep=False, seed=-1):
    """Get fitness of a single individual.
  
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      nReps   - (nReps)    - number of trials to get average fitness
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - mean reward over all trials
    """
    if nRep is False:
      nRep = self.nReps
    wVec[np.isnan(wVec)] = 0
    reward = np.empty(nRep)
    for iRep in range(nRep):
      if seed > 0:
        seed = seed+iRep
      reward[iRep] = self.testInd(wVec, aVec, view=view, seed=seed)
    fitness = np.mean(reward)
    return fitness

  def testInd(self, wVec, aVec, alg_critic, mem, batch_size=1024,
              view=False, seed=-1, mem_sample=True,
              bootstrap_default=-100.0): # TODO: make bootstrap default config driven
    """Evaluate individual on task
    Args:
      wVec    - (np_array) - weight matrix as a flattened vector
                [N**2 X 1]
      aVec    - (np_array) - activation function of each node 
                [N X 1]    - stored as ints (see applyAct in ann.py)
  
    Optional:
      view    - (bool)     - view trial?
      seed    - (int)      - starting random seed for trials
  
    Returns:
      fitness - (float)    - reward earned in trial
    """
    if seed >= 0:
      random.seed(seed)
      np.random.seed(seed)
      self.env.seed(seed)

    state = self.env.reset()
    self.env.t = 0
    annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
    action = selectAct(annOut, self.actSelect)

    # TODO: replace emulator step with critic eval from SAC here

    state, reward, done, info = self.env.step(action)

    if mem_sample:
      if mem is None or mem.size() < batch_size:
        ret = bootstrap_default
      else:
        batch = mem.sample(batch_size=batch_size)
        ret = alg_critic(batch.observations, batch.actions)[0].mean().item()
    else:
      if self.maxEpisodeLength == 0:
        if view:
          if self.needsClosed:
            self.env.render(close=done)
          else:
            self.env.render()
        return reward
      else:
        totalReward = reward

      if alg_critic is None:
        n_episodes = self.maxEpisodeLength
      else:
        n_episodes = 5  # TODO: make n_boostrap steps config driven

      gamma = .99
      rewards = []
      for tStep in range(n_episodes):
        annOut = act(wVec, aVec, self.nInput, self.nOutput, state)
        action = selectAct(annOut,self.actSelect)

        state, reward, done, info = self.env.step(action)
        if tStep == self.maxEpisodeLength-1 and not done:
          if alg_critic is not None:
            rewards.append(alg_critic(state, action))
          break
        else:
          rewards.append(reward)
        if view:
          if self.needsClosed:
            self.env.render(close=done)
          else:
            self.env.render()
        if done:
          break

      ret = None
      if alg_critic is None:
        ret = np.sum(rewards)
      else:
        d = np.array([gamma**i for i in range(n_episodes+2)])
        ret = 0.0
        for i, _ in enumerate(rewards):
          ret += sum(rewards[i:]*d[:-(1+i)])

    return ret


def discount_rewards(self, rs):
  drs = np.zeros_like(rs).asnumpy()
  s = 0
  for t in reversed(xrange(0, len(rs))):
    # Reset the running sum at a game boundary.
    if rs[t] != 0:
      s = 0
    s = s * self.gamma + rs[t]
    drs[t] = s
  drs -= np.mean(drs)
  drs /= np.std(drs)
  return drs
