import multiprocessing as mp
import gym
import numpy as np
import copy
from task import task
import os
import config
from stable_baselines3.sac import MlpPolicy

# TODO: add yml config binding
ENV_NAME = 'CartPole-v1'


def get_task_config():
    wann_param_config = task.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME
    wann_param_config['maxGen'] = 1
    wann_param_config['popSize'] = 20
    wann_param_config['alg_nReps'] = 1

    task_config = dict(
        WANN_ENV_ID='wann-cartpole',
        NUM_WORKERS=5,  # mp.cpu_count() for all cores
        GAME_CONFIG=task.Game(env_name='CartPole-v1',
                              actionSelect='all',  # all, soft, hard
                              input_size=4,  # greyscale + 4 frame stacking
                              output_size=4,
                              time_factor=0,
                              layers=[25, 25],
                              i_act=np.full(4, 1),
                              h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              o_act=np.full(4, 1),
                              weightCap=2.0,
                              noise_bias=0.0,
                              output_noise=[False, False, False],
                              max_episode_length=10,  # use full episode length or reasonable trajectory len here
                              n_critic_bootstrap=5,
                              alg_type=task.ALG.SAC,
                              artifacts_path=f'{task.RESULTS_PATH}artifact{os.sep}{config.EXPERIMENT_ID}{os.sep}',
                              in_out_labels=[]),
        AGENT=dict(
            datprep=None,
            policy=MlpPolicy,
            learn_params=dict(
                gamma=0.99,
                tau=5e-3,
                learn_rate=1e-4,
                mem_size=int(1E6),
                target_entropy='auto',
                timesteps=300,  # for baseline SAC give enough timesteps to be at least 500 episodes
                train_batch_size=100,  # batch buffer size
                episode_len=-1,  # entire episode length
                eval_episode_len=-1,
                alg_checkpoint_interval=500,
                start_steps=10,
                n_trains_per_step=1,  # soft target updates should use 1, try 5 for hard target updates
                gradient_steps_per_step=1,
                eval_interval=1500,
                log_interval=10,
                log_verbose=1,
                replay_sample_ratio=1,  # 1:1 or 4:1 for 1e6 replay mem size
            )
        ),
        ENTRY_POINT='task.cart_pole:_env',
        WANN_PARAM_CONFIG=wann_param_config,
        VIDEO_LENGTH=1500,
        RESULTS_PATH=task.RESULTS_PATH
    )

    return copy.deepcopy(task_config)


def _env():
    if task.WANN_OUT_PREFIX is None:
        raise Exception('WANN out prefix was not set')

    env = task.ObsWrapper(gym.make(ENV_NAME),
                          champion_artifacts_path=f'{task.WANN_OUT_PREFIX}_best.out')
    return env
