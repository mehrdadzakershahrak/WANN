import random
import numpy as np
import sys
from extern.wann.domain.make_env import make_env
from extern.wann.neat_src import *
from stable_baselines import PPO2, DDPG
from task import task
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
tf.get_logger().setLevel('FATAL')


class GymTask():
  """Problem domain to be solved by neural network. Uses OpenAI Gym patterns.
  """ 
  def __init__(self, game, paramOnly=False, nReps=1,
               agent_params=None, agent_env=None):
    """Initializes task environment
  
    Args:
      game - (string) - dict key of task to be solved (see domain/config.py)
  
    Optional:
      paramOnly - (bool)  - only load parameters instead of launching task?
      nReps     - (nReps) - number of trials to get average fitness
    """

    if agent_params is None:
      raise Exception('AGENT PARAMS IS NONE')

    if agent_params is None:
      raise Exception('AGENT ENV IS NONE')

    # Network properties
    self.nInput   = game.input_size
    self.nOutput  = game.output_size      
    self.actRange = game.h_act
    self.absWCap  = game.weightCap
    self.layers   = game.layers
    self.activations = np.r_[np.full(1,1),game.i_act,game.o_act]

    # TODO: clean this up
    # HACK

    self.alg      = game.alg
    if self.alg == task.ALG.PPO:
      # TODO: make network policy config driven

      agent = PPO2(MlpPolicy, agent_env, verbose=0) # TODO: integrate with MPI from top level for performance
      agent.load_parameters(agent_params)

      self.agent = agent
    elif self.alg == task.ALG.DDPG:
      n_actions = agent_env.action_space.shape[-1]
      action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                  sigma=float(0.5) * np.ones(n_actions))
      agent = DDPG(DDPG_MlpPolicy, agent_env, verbose=0, param_noise=None, action_noise=action_noise)
      agent.load_parameters(agent_params)
    elif self.alg == task.ALG.TD3:
      pass
    else:
      raise Exception(f'Algorithm configured is not currently supported')

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

  def testInd(self, wVec, aVec, hyp=None, view=False,seed=-1):
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

    # TODO: add passthrough for deterministic vs stochastic output

    with tf.device('/cpu:0'):
      action, state = self.agent.predict(annOut)
      action = np.array(action)[0]

    # previous prediction:
    # action = selectAct(annOut,self.actSelect)

    state, reward, done, info = self.env.step(action)
    
    if self.maxEpisodeLength == 0:
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      return reward
    else:
      totalReward = reward
    
    for tStep in range(self.maxEpisodeLength): 
      annOut = act(wVec, aVec, self.nInput, self.nOutput, state) 
      action = selectAct(annOut,self.actSelect)

      with tf.device('/cpu:0'):
        action, state = self.agent.predict(annOut)
        action = np.array(action)[0]

      state, reward, done, info = self.env.step(action)
      totalReward += reward  
      if view:
        if self.needsClosed:
          self.env.render(close=done)  
        else:
          self.env.render()
      if done:
        break
    return totalReward
