from collections import namedtuple
from extern.wann import wann_test as wtest
from extern.wann import wann_train as wtrain
from extern.wann.neat_src import ann as wnet
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines import PPO2
import gym
import numpy as np
import os
import multiprocessing as mp
import config

# TODO: clean this up

ARTIFACTS_PATH = f'{config.RESULTS_PATH}{os.sep}wann-ppo2-model'
WANN_CHAMPION_PATH = f'champion{os.sep}'

NUM_WORKERS = mp.cpu_count()


def balance():
    base_env = 'CartPole-v1'
    # See reference to WANN extern wann/domain/config.py for reference config

    setup_env = gym.make(base_env)
    setup_obs = setup_env.reset()

    # TODO: make env config driven
    cartpole_balance = config.Game(env_name=base_env,
                            actionSelect='all',  # all, soft, hard
                            input_size=setup_obs.shape[0],
                            output_size=setup_env.action_space.n,
                            time_factor=0,
                            layers=[setup_obs.shape[0], setup_obs.shape[0]],
                            i_act=np.full(setup_obs.shape[0], setup_env.action_space.n),
                            h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            o_act=np.full(setup_env.action_space.n, setup_env.action_space.n),
                            weightCap=2.0,
                            noise_bias=0.0,
                            output_noise=[False, False, False],
                            max_episode_length=200,
                            in_out_labels=['x', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot',
                                           'force']
                            )
    del setup_env
    del setup_obs

    games = {
        base_env: cartpole_balance

    }

    wtrain.init_games_config(games)

    # TODO: add flg here to determine if pre-training is needed
    # Train WANN feature extractor
    eid = 'wann-cartpolebalance-v1'
    gym.envs.register(
        id=eid,
        entry_point='task.cartpole:_balance_env',
        max_episode_steps=30000
    )

    wann_param_config = config.get_default_wann_hyperparams()
    wann_param_config['task'] = base_env

    # TODO: add timestamp here to keep multi results
    outPrefix = config.RESULTS_PATH+eid

    wann_args = dict(
        hyperparam=wann_param_config,
        outPrefix=outPrefix,
        num_workers=mp.cpu_count(),
        games=games
    )

    if config.SHOULD_TRAIN_WANN:
        wtrain.run(wann_args)

    # TODO: add in champion selection here
    # TODO: match up training size

    env = make_vec_env(eid, n_envs=mp.cpu_count())

    m = PPO2(MlpPolicy, env, verbose=1)
    m.learn(total_timesteps=2000000, log_interval=10)

    m.save(ARTIFACTS_PATH)

    m = PPO2.load(ARTIFACTS_PATH)

    # wVec, aVec, _ = wnet.importNet(f'log{os.sep}test_best{os.sep}0008.out')
    # feats = wnet.act(wVec, aVec[:-2],
    #                  nInput=obs.shape[1],
    #                  nOutput=obs.shape[1],
    #                  inPattern=obs)

    test_env = gym.make(eid)
    obs = test_env.reset()
    while True:
        a, s = m.predict(obs, deterministic=True)
        obs, r, done, _ = test_env.step(a)
        test_env.render(mode='human')

    print('test')


def _balance_env():
    # TODO: replace with actual model artifacts config path
    env = CartPoleObsWrapper(gym.make('CartPole-v1'),
                             m_artifacts_path='extern/wann/champions/swing.out')
    return env


class CartPoleObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, m_artifacts_path):
        super().__init__(env)

        self.wVec, self.aVec, _ = wnet.importNet(m_artifacts_path)
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


def update_champion():
    pass
