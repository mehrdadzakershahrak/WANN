from task_config import cartpole
from extern.wann import wann_train as wtrain
from extern.wann.neat_src import ann as wnet
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
import gym
import os
import multiprocessing as mp
import config
import extern.wann.vis as wann_vis
import matplotlib.pyplot as plt
from task_config.task_config import ALG


SEED_RANGE_MIN = 1
SEED_RANGE_MAX = 100000000


# TODO: proper logging
def main():
    if config.TASK in ['cartpole-balance']:
        run(cartpole.get_task_config())
    if config.TASK in ['bipedal-walker']:
        run(cartpole.get_task_config())
    else:
        raise Exception('No implemented environment found. Please refer to list of implemented environments in README')


def run(config):
    ARTIFACTS_PATH = f'{RESULTS_PATH}artifact{os.sep}{config.EXPERIMENT_ID}{os.sep}'
    VIS_RESULTS_PATH = f'{RESULTS_PATH}vis{os.sep}{config.EXPERIMENT_ID}{os.sep}'
    SAVE_GIF_PATH = f'{RESULTS_PATH}gif{os.sep}'
    TB_LOG_PATH = f'{RESULTS_PATH}tb-log{os.sep}{config.EXPERIMENT_ID}{os.sep}'
    WANN_OUT_PREFIX = f'{RESULTS_PATH}wann-run{os.sep}{config.EXPERIMENT_ID}{os.sep}'

    paths = [ARTIFACTS_PATH, VIS_RESULTS_PATH, TB_LOG_PATH]
    for p in paths:
        if not os.path.isdir(p):
            os.makedirs(p)

    GAME_CONFIG = config['GAME_CONFIG']
    EXPERIMENT_ID = GAME_CONFIG['EXPERIMENT_ID']
    ENV_NAME = GAME_CONFIG.env_name

    games = {
        ENV_NAME: GAME_CONFIG
    }

    wtrain.init_games_config(games)
    gym.envs.register(
        id=EXPERIMENT_ID,
        entry_point=config['ENTRY_POINT'],
        max_episode_steps=GAME_CONFIG['max_episode_length']
    )

    wann_param_config = config['WANN_PARAM_CONFIG']
    wann_args = dict(
        hyperparam=wann_param_config,
        outPrefix=WANN_OUT_PREFIX,
        num_workers=mp.cpu_count(),
        games=games
    )

    if config.SHOULD_TRAIN_WANN:
        wtrain.run(wann_args)

    if config.SHOULD_VISUALIZE_WANN:
        champion_path = f'{ARTIFACTS_PATH}{EXPERIMENT_ID}_best.out'
        wVec, aVec, _ = wnet.importNet(champion_path)

        wann_vis.viewInd(champion_path, GAME_CONFIG)
        plt.savefig(f'{VIS_RESULTS_PATH}wann-net-graph.png')

    m = None
    if not config.USE_PREV_EXPERIMENT:
        agent_config = config['AGENT']
        alg = config['ALG']
        if alg == ALG.PPO:
            env = make_vec_env(ENV_NAME, n_envs=mp.cpu_count())
            m = PPO2(MlpPolicy, env, verbose=agent_config['verbose'], tensorboard_log=TB_LOG_PATH)
            m.learn(total_timesteps=agent_config['total_timesteps'], log_interval=agent_config['log_interval'])
            m.save(ARTIFACTS_PATH)
            m = PPO2.load(ARTIFACTS_PATH)
        elif alg == ALG.DDPG:
            pass
        elif alg == ALG.TD3:
            pass
        else:
            raise Exception(f'Algorithm configured is not currently supported')
    else:
        m = PPO2.load(config.PREV_EXPERIMENT_PATH)

    # TODO: additional test visualizations here


if __name__ == '__main__':
    main()
