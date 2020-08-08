from extern.wann import wann_train as wtrain
from extern.wann.neat_src import ann as wnet
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import gym
import numpy as np
import os
import multiprocessing as mp
import config
from vis import plot
import random

# TODO: clean this up

ARTIFACTS_PATH = f'{config.RESULTS_PATH}wann-ppo2-model{config.EXPERIMENT_ID}{os.sep}'
VIS_RESULTS_PATH = f'{ARTIFACTS_PATH}{os.sep}{config.EXPERIMENT_ID}{os.sep}vis{os.sep}'
eid = None
NUM_WORKERS = mp.cpu_count()

TB_LOG_PATH = f'log{os.sep}wann-ppo2-model{os.sep}{config.EXPERIMENT_ID}{os.sep}'


# TODO: update to auto save multiple experiments / reload most recent
def balance():
    global eid

    base_env = 'CartPole-v1'

    setup_env = gym.make(base_env)
    setup_obs = setup_env.reset()

    cartpole_balance = config.Game(env_name=base_env,
                            actionSelect='prob',
                            input_size=setup_obs.shape[0],
                            output_size=setup_env.action_space.n,
                            time_factor=0,
                            layers=[setup_obs.shape[0], setup_obs.shape[0]],
                            i_act=np.full(setup_obs.shape[0], setup_obs.shape[0]),
                            h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                            o_act=np.full(setup_obs.shape[0], 1),
                            weightCap=2.0,
                            noise_bias=0.0,
                            output_noise=[False, False, False],
                            max_episode_length=30000,
                            in_out_labels=['x', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot',
                                           'force'])
    del setup_env
    del setup_obs

    games = {
        base_env: cartpole_balance

    }

    wtrain.init_games_config(games)

    eid = 'wann-cartpolebalance-v1'
    gym.envs.register(
        id=eid,
        entry_point='task.cartpole:_balance_env',
        max_episode_steps=30000
    )

    wann_param_config = config.get_default_wann_hyperparams()
    wann_param_config['task'] = base_env
    wann_param_config['maxGen'] = 5

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

    env = make_vec_env(eid, n_envs=mp.cpu_count())

    if not config.USE_PREV_EXPERIMENT:
        m = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=TB_LOG_PATH)
        m.learn(total_timesteps=2000000, log_interval=10)
        m.save(ARTIFACTS_PATH)
        m = PPO2.load(ARTIFACTS_PATH)
    else:
        m = PPO2.load(config.PREV_EXPERIMENT_PATH)

    test_env = gym.make(eid)
    obs = test_env.reset()

    avg_rewards = []
    scores = []
    episodes = 10000
    t_len = 30000
    for i in range(episodes):
        t = 0
        rewards = []

        test_env.seed(random.randint(config.SEED_RANGE_MIN, config.SEED_RANGE_MAX))
        obs = test_env.reset()
        for _ in range(t_len):
            a, s = m.predict(obs, deterministic=True)
            a = np.array(a).item()  # workaround for [a] return value
            obs, r, done, _ = test_env.step(a)

            if done:
                break

            if config.SHOW_TESTS:
                test_env.render(mode='human')

            rewards.append(r)
            t += 1

        avg_rewards.append(np.array(rewards)/float(t))
        scores.append(t)

        if i % 10000 == 0:
            print(f'Completed episode: {i}')

    # TODO performance comparison of multiple RL algos here


def _balance_env():
    env = CartPoleObsWrapper(gym.make('CartPole-v1'),
                             champion_artifacts_path=config.RESULTS_PATH+eid+'_best.out')
    return env


class CartPoleObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, champion_artifacts_path):
        super().__init__(env)

        self.wVec, self.aVec, _ = wnet.importNet(champion_artifacts_path)

    def observation(self, obs):
        if config.SHOULD_USE_WANN:
            obs = wnet.act(self.wVec, self.aVec,
                             nInput=obs.shape[0],
                             nOutput=obs.shape[0],
                             inPattern=obs)
        return obs
