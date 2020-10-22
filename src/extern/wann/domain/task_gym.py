import random
import numpy as np
import sys
from extern.wann.domain.make_env import make_env
from extern.wann.neat_src import *
from stable_baselines3 import PPO, DDPG
from task import task
import torch as th
from extern.wann.neat_src import ann as wnet
from extern.wann import wann_train as wtrain


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

  def testInd(self, wVec, aVec, alg_critic, alg_policy,
              mem, batch_size=None,
              view=False, seed=-1, bootstrap_default=None): # TODO: make bootstrap default config driven
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
    if mem is None or mem.size() < batch_size:
      ret = bootstrap_default
    else:
      _, wann_obs, *_ = mem.raw_sample(batch_size)

      n_feats = wann_obs.shape[1]

      # TODO: flatten obs for CNN

      with th.no_grad():
        obs_batch = []
        for o in wann_obs:
          obs_batch.append(wnet.act(wVec, aVec,
                                    nInput=n_feats,
                                    nOutput=n_feats,
                                    inPattern=o))
        obs_batch = th.from_numpy(np.array(obs_batch)).to(wtrain.DEVICE)

        acts = alg_policy(obs_batch)
        ret = alg_critic(obs_batch, acts)[0].mean().item()

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
