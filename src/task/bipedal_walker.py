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
# EXPERIMENT_ID = 'wann-bipedalwalker2-v2'
ENV_NAME = 'BipedalWalker-v3'
WANN_OUT_PREFIX = f'{task.RESULTS_PATH}{config.EXPERIMENT_ID}{os.sep}artifact{os.sep}wann{os.sep}'


def get_task_config():
    setup_env = gym.make(ENV_NAME)
    setup_obs = setup_env.reset()

    wann_param_config = task.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME
    wann_param_config['maxGen'] = 5  # TODO: 20 - 4 steps
    wann_param_config['popSize'] = 20  # TODO: 64 - 5 steps
    wann_param_config['alg_nReps'] = 1

    task_config = dict(
        WANN_ENV_ID='wann-bipedalwalker-v3', # THIS IS ACTUALLY DIFFERENT THAT EXPERIMENT ID DUE TO GEN X EXPERIMENT CYCLES
        NUM_WORKERS=mp.cpu_count(),
        GAME_CONFIG=task.Game(env_name='BipedalWalker-v3',
                  actionSelect='all',  # all, soft, hard
                  input_size=24,
                  output_size=24,
                  time_factor=0,
                  layers=[40, 40],
                  i_act=np.full(24,1),
                  h_act=[1,2,3,4,5,6,7,8,9,10],
                  o_act=np.full(4,1),
                  weightCap=2.0,
                  noise_bias=0.0,
                  output_noise=[False, False, False],
                  max_episode_length=1600,  # n_step_boostrap - total episode length 1600
                  n_critic_bootstrap=200,
                  alg_type=task.ALG.SAC,
                  artifacts_path=f'{task.RESULTS_PATH}artifact{os.sep}{config.EXPERIMENT_ID}{os.sep}',
                  in_out_labels=[
                  'hull_angle','hull_vel_angle','vel_x','vel_y',
                  'hip1_angle','hip1_speed','knee1_angle','knee1_speed','leg1_contact',
                  'hip2_angle','hip2_speed','knee2_angle','knee2_speed','leg2_contact',
                  'lidar_0','lidar_1','lidar_2','lidar_3','lidar_4',
                  'lidar_5','lidar_6','lidar_7','lidar_8','lidar_9',
                  'hip_1','knee_1','hip_2','knee_2']),
        AGENT=dict(
            learn_params=dict(
                gamma=0.99,
                tau=5e-3,  # TODO: also try 5e-3 from the literature
                learn_rate=1e-4,
                mem_size=int(1E6),
                target_entropy='auto',
                # n_hidden=256,
                # n_depth=2,
                # clip_val=1.0,
                # alpha=.2,
                # n_epochs=300,
                # policy_lr=1e-4,  # TODO: try 3e-3 and 1e-3
                # qf_lr=1e-4,  # TODO: try 3e-3 and 1e-3
                # reward_scale=1.0,  # TODO: also try 5 for walker from literature
                timesteps=3200,  # episode length is 1600 e.g. x 10 = 16000 timesteps Example SAC full run to converge: 420000
                train_batch_size=100,  # batch buffer size
                episode_len=-1,  # entire episode length
                eval_episode_len=-1,
                alg_checkpoint_interval=1000,
                start_steps=100,
                n_trains_per_step=1,  # soft target updates should use 1, try 5 for hard target updates
                gradient_steps_per_step=1,
                eval_interval=2000,
                log_interval=10,
                log_verbose=1,
                replay_sample_ratio=1,  # 4:1 or .25 replay buffer sample to gradient update ratio
            )
        ),
        ENTRY_POINT='task.bipedal_walker:_env',
        WANN_PARAM_CONFIG=wann_param_config,
        VIDEO_LENGTH=1600,
        RESULTS_PATH=task.RESULTS_PATH
    )

    del setup_env
    del setup_obs

    return copy.deepcopy(task_config)


def _env():
    env = task.ObsWrapper(gym.make(ENV_NAME),
                          champion_artifacts_path=f'{WANN_OUT_PREFIX}_best.out')
    return env
