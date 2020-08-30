import multiprocessing as mp
import gym
import numpy as np
import copy
from task import task
import os
import config

# TODO: add yml config binding
# TODO: simplify this further

# TODO: add generated date timestamp for unique experiment id cycled by x experiments
# EXPERIMENT_ID = 'wann-bipedalwalker2-v3'
ENV_NAME = 'BipedalWalker-v3'
WANN_OUT_PREFIX = f'{task.RESULTS_PATH}{config.EXPERIMENT_ID}{os.sep}wann{os.sep}'


def get_task_config():
    setup_env = gym.make(ENV_NAME)
    setup_obs = setup_env.reset()

    wann_param_config = task.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME

    task_config = dict(
        WANN_ENV_ID='wann-bipedalwalker-v3', # THIS IS ACTUALLY DIFFERENT THAT EXPERIMENT ID DUE TO GEN X EXPERIMENT CYCLES
        NUM_WORKERS=mp.cpu_count(),
        GAME_CONFIG=task.Game(env_name='BipedalWalker-v3',
                  actionSelect='all', # all, soft, hard
                  input_size=24,
                  output_size=24,
                  time_factor=0,
                  layers=[40, 40],
                  i_act=np.full(24,4),
                  h_act=[1,2,3,4,5,6,7,8,9,10],
                  o_act=np.full(4,4),
                  weightCap=2.0,
                  noise_bias=0.0,
                  output_noise=[False, False, False],
                  max_episode_length=400,
                  alg=task.ALG.PPO,
                  artifacts_path=f'{task.RESULTS_PATH}artifact{os.sep}{config.EXPERIMENT_ID}{os.sep}',
                  in_out_labels=[
                  'hull_angle','hull_vel_angle','vel_x','vel_y',
                  'hip1_angle','hip1_speed','knee1_angle','knee1_speed','leg1_contact',
                  'hip2_angle','hip2_speed','knee2_angle','knee2_speed','leg2_contact',
                  'lidar_0','lidar_1','lidar_2','lidar_3','lidar_4',
                  'lidar_5','lidar_6','lidar_7','lidar_8','lidar_9',
                  'hip_1','knee_1','hip_2','knee_2']),
        AGENT=dict(
            verbose=1,
            log_interval=5,
            total_timesteps=1000
        ),
        ENTRY_POINT='task.bipedal_walker:_env',
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
