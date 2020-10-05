import gym
import numpy as np
import copy
from task import task
import os
import config
from stable_baselines3.sac import MlpPolicy

# TODO: add yml config binding
ENV_NAME = 'HumanoidPyBulletEnv-v0'
WANN_OUT_PREFIX = f'{task.RESULTS_PATH}{config.EXPERIMENT_ID}{os.sep}artifact{os.sep}wann{os.sep}'


def get_task_config():
    wann_param_config = task.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME
    wann_param_config['maxGen'] = 1
    wann_param_config['popSize'] = 100
    wann_param_config['alg_nReps'] = 1

    task_config = dict(
        WANN_ENV_ID='wann-humanoid-v0',
        NUM_WORKERS=5,
        DEVICE='cuda:0',
        GAME_CONFIG=task.Game(env_name=ENV_NAME,
                              actionSelect='all',  # OPTIONS: soft, all, hard
                              input_size=44,
                              output_size=44,
                              time_factor=0,
                              layers=[40, 40],
                              i_act=np.full(44, 1),
                              h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              o_act=np.full(44, 1),
                              weightCap=2.0,
                              noise_bias=0.0,
                              output_noise=[False, False, False],
                              max_episode_length=500,
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
                mem_size=int(1e6),
                target_entropy='auto',
                timesteps=1000,
                train_batch_size=100,  # batch buffer size
                episode_len=-1,  # entire episode length
                eval_episode_len=-1,
                alg_checkpoint_interval=500,
                start_steps=100,
                n_trains_per_step=1,  # soft target updates should use 1, try 5 for hard target updates
                gradient_steps_per_step=1,
                eval_interval=1500,
                log_interval=10,
                log_verbose=1,
                replay_sample_ratio=1,  # 4:1 or .25 replay buffer sample to gradient update ratio
            )
        ),
        ENTRY_POINT='task.humanoid:_env',
        WANN_PARAM_CONFIG=wann_param_config,
        VIDEO_LENGTH=1500,
        RESULTS_PATH=task.RESULTS_PATH
    )

    return copy.deepcopy(task_config)


def _env():
    env = task.ObsWrapper(gym.make(ENV_NAME),
                          champion_artifacts_path=f'{WANN_OUT_PREFIX}_best.out')
    return env
