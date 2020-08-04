from collections import namedtuple
from extern.wann import wann_test as wtest
from extern.wann import wann_train as wtrain
from extern.wann.neat_src import ann as wnet
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import ACKTR
import gym
import numpy as np
import os
import multiprocessing as mp

# TODO: clean this up

NUM_WORKERS = mp.cpu_count()
Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
                           'input_size', 'output_size', 'layers', 'i_act', 'h_act',
                           'o_act', 'weightCap', 'noise_bias', 'output_noise',
                           'max_episode_length', 'in_out_labels'])
games = {}

# See reference to WANN extern wann/domain/config.py for reference config
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
    actionSelect='all', # all, soft, hard
    input_size=5,
    output_size=1,
    time_factor=0,
    layers=[5, 5],
    i_act=np.full(5,1),
    h_act=[1,2,3,4,5,6,7,8,9,10],
    o_act=np.full(1,1),
    weightCap=2.0,
    noise_bias=0.0,
    output_noise=[False, False, False],
    max_episode_length=200,
    in_out_labels=['x', 'x_dot','cos(theta)','sin(theta)','theta_dot',
                   'force']
)

games['swingup'] = cartpole_swingup


def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        gym.envs.register(
            id=env_id,
            entry_point='task.cartpole:_balance_env',
            max_episode_steps=10000
        )

        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init


def balance(args):
    wtrain.init_games_config(games)

    # TODO: add flg here to determine if pre-training is needed
    # Train WANN feature extractor
    eid = 'wann-cartpolebalance-v1'

    gym.envs.register(
        id=eid,
        entry_point='task.cartpole:_balance_env',
        max_episode_steps=10000
    )
    # wtrain.run(args)

    # TODO: add in champion selection here
    # TODO: match up training size

    env = SubprocVecEnv([make_env(eid, i) for i in range(NUM_WORKERS)])
    m = ACKTR(MlpPolicy, env, verbose=1)
    m.learn(total_timesteps=30000)

    obs = env.reset()

    wVec, aVec, _ = wnet.importNet(f'log{os.sep}test_best{os.sep}0008.out')
    feats = wnet.act(wVec, aVec[:-2],
                     nInput=obs.shape[1],
                     nOutput=obs.shape[1],
                     inPattern=obs)

    while True:
        a, s = m.predict(obs)
        obs, r, dones, _ = env.step(a)
        env.render()

    print('test')


def _balance_env():
    # TODO: replace with actual model artifacts config path
    env = CartPoleObsWrapper(gym.make('CartPole-v1'),
                             m_artifacts_path='log/wann/champio')
    return env


class CartPoleObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, m_artifacts_path):
        super().__init__(env)

        self.wVec, self.aVec, _ = wnet.importNet('extern/wann/champions/swing.out')
        # self.n_obs_space = n_obs_space  TODO: pull dynamic
        # self.wVec[:-self.n_obs_space+1] = 1.0

    def observation(self, obs):
        # modify obs

        # TODO add wann feature extraction here
        # print('OBSERVATION CALLED')

        # feats = wnet.act(self.wVec, self.aVec[:-2],
        #                  nInput=self.n_obs_space,
        #                  nOutput=self.n_obs_space,
        #                  pattern=obs)

        return obs
