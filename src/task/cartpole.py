import multiprocessing as mp
import gym
import numpy as np
import copy
from task import task
import os

# TODO: add yml config binding
# TODO: simplify this further

# TODO: add generated date timestamp for unique experiment id cycled by x experiments
EXPERIMENT_ID = 'wann-cartpolebalance-v1'
ENV_NAME = 'CartPole-v1'
WANN_OUT_PREFIX = f'{task.RESULTS_PATH}{EXPERIMENT_ID}{os.sep}wann{os.sep}'


def get_task_config():
    setup_env = gym.make(ENV_NAME)
    setup_obs = setup_env.reset()

    wann_param_config = task.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME

    task_config = dict(
        WANN_ENV_ID='wann-cartpolebalance-v1', # THIS IS ACTUALLY DIFFERENT THAT EXPERIMENT ID DUE TO GEN X EXPERIMENT CYCLES
        NUM_WORKERS=mp.cpu_count(),
        GAME_CONFIG=task.Game(env_name=ENV_NAME,
                              actionSelect='prob',
                              input_size=setup_obs.shape[0],
                              output_size=setup_obs.shape[0],
                              time_factor=0,
                              layers=[setup_obs.shape[0], setup_obs.shape[0]],
                              i_act=np.full(setup_obs.shape[0], setup_obs.shape[0]),
                              h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              o_act=np.full(setup_obs.shape[0], setup_obs.shape[0]),
                              weightCap=2.0,
                              noise_bias=0.0,
                              output_noise=[False, False, False],
                              max_episode_length=30000,
                              alg=task.ALG.PPO,
                              artifacts_path=f'{task.RESULTS_PATH}artifact{os.sep}{EXPERIMENT_ID}{os.sep}',
                              in_out_labels=['x', 'x_dot', 'cos(theta)', 'sin(theta)', 'theta_dot',
                                             'force']),
        MAX_EPISODE_STEPS=30000,
        AGENT=dict(
            verbose=1,
            log_interval=100,
            total_timesteps=30000
        ),
        ENTRY_POINT='task.cartpole:_env',
        EXPERIMENT_ID=EXPERIMENT_ID,
        WANN_PARAM_CONFIG=wann_param_config,
        VIDEO_LENGTH=1500,
        RESULTS_PATH=task.RESULTS_PATH
    )

    del setup_env
    del setup_obs

    return copy.deepcopy(task_config)


def _env():
    env = task.ObsWrapper(gym.make(ENV_NAME),
                          champion_artifacts_path=f'{WANN_OUT_PREFIX}_best.out')
    return env
