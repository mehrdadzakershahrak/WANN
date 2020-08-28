import multiprocessing as mp
import config
import gym
import numpy as np
import copy
from task_config import task_config
import os

# TODO: add yml config binding
# TODO: simplify this further

# TODO: add generated date timestamp for unique experiment id cycled by x experiments
EXPERIMENT_ID = 'wann-cartpole'
ENV_NAME = 'CartPole-v2'
WANN_OUT_PREFIX = f'{task_config.RESULTS_PATH}wann-run{os.sep}{EXPERIMENT_ID}{os.sep}'


def get_task_config():
    setup_env = gym.make(ENV_NAME)
    setup_obs = setup_env.reset()

    wann_param_config = task_config.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME
    task_config = dict(
        NUM_WORKERS=mp.cpu_count(),
        GAME_CONFIG=config.Game(env_name=ENV_NAME,
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
                                               'force']),
        MAX_EPISODE_STEPS=30000,
        AGENT_CONFIG=dict(
            verbose=1
        ),
        ENTRY_POINT='task.cartpole:_env',
        EXPERIMENT_ID=EXPERIMENT_ID,
        WANN_PARAM_CONFIG=wann_param_config,
        ALG=task_config.ALG.PPO
    )

    del setup_env
    del setup_obs

    return copy.deepcopy(task_config)


def _env():
    env = task_config.ObsWrapper(gym.make(ENV_NAME),
                                 champion_artifacts_path=f'{WANN_OUT_PREFIX}{EXPERIMENT_ID}_best.out')
    return env
