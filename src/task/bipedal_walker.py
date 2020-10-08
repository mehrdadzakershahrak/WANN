import numpy as np
import copy
from task import task
import os
import config
from stable_baselines3.sac import MlpPolicy

# TODO: add yml config binding
ENV_NAME = 'BipedalWalker-v3'


def get_task_config():
    wann_param_config = task.get_default_wann_hyperparams()
    wann_param_config['task'] = ENV_NAME
    wann_param_config['maxGen'] = 1
    wann_param_config['popSize'] = 100
    wann_param_config['alg_nReps'] = 1

    task_config = dict(
        WANN_ENV_ID='wann-bipedalwalker-v3',
        NUM_WORKERS=5,
        DEVICE='cuda:0',
        GAME_CONFIG=task.Game(env_name=ENV_NAME,
                              actionSelect='all',  # OPTIONS: soft, all, hard
                              input_size=24,
                              output_size=24,
                              time_factor=0,
                              layers=[40, 40],
                              i_act=np.full(24, 1),
                              h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                              o_act=np.full(24, 1),
                              weightCap=2.0,
                              noise_bias=0.0,
                              output_noise=[False, False, False],
                              max_episode_length=1600,
                              n_critic_bootstrap=5,
                              alg_type=task.ALG.SAC,
                              artifacts_path=f'{task.RESULTS_PATH}artifact{os.sep}{config.EXPERIMENT_ID}{os.sep}',
                              in_out_labels=[
                                  'hull_angle', 'hull_vel_angle', 'vel_x', 'vel_y',
                                  'hip1_angle', 'hip1_speed', 'knee1_angle', 'knee1_speed', 'leg1_contact',
                                  'hip2_angle', 'hip2_speed', 'knee2_angle', 'knee2_speed', 'leg2_contact',
                                  'lidar_0', 'lidar_1', 'lidar_2', 'lidar_3', 'lidar_4',
                                  'lidar_5', 'lidar_6', 'lidar_7', 'lidar_8', 'lidar_9',
                                  'hip_1', 'knee_1', 'hip_2', 'knee_2']),
        AGENT=dict(
            datprep=None,
            policy=MlpPolicy,
            learn_params=dict(
                gamma=0.99,
                tau=5e-3,
                learn_rate=1e-4,
                mem_size=int(1e6),
                target_entropy='auto',
                timesteps=1600,
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
        WANN_PARAM_CONFIG=wann_param_config,
        VIDEO_LENGTH=1500,
        RESULTS_PATH=task.RESULTS_PATH
    )

    return copy.deepcopy(task_config)
