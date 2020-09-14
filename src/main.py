from task import cartpole, bipedal_walker
from extern.wann import wann_train as wtrain
from extern.wann.neat_src import ann as wnet
import gym
import os
import extern.wann.vis as wann_vis
import matplotlib.pyplot as plt
from task import task
import imageio
import numpy as np
import config as run_config
import sys
from mpi4py import MPI
import subprocess
import pickle
from agent import sac as alg

from rlkit.data_management.replay_buffer import ReplayBuffer

comm = MPI.COMM_WORLD
rank = 0

SEED_RANGE_MIN = 1
SEED_RANGE_MAX = 100000000
LOG_INTERVAL = 10


def run(config):
    RESULTS_PATH = config['RESULTS_PATH']
    ARTIFACTS_PATH = f'{RESULTS_PATH}artifact{os.sep}{run_config.EXPERIMENT_ID}{os.sep}'
    VIS_RESULTS_PATH = f'{RESULTS_PATH}vis{os.sep}{run_config.EXPERIMENT_ID}{os.sep}'
    SAVE_GIF_PATH = f'{RESULTS_PATH}gif{os.sep}'
    TB_LOG_PATH = f'{RESULTS_PATH}tb-log{os.sep}{run_config.EXPERIMENT_ID}{os.sep}'
    WANN_OUT_PREFIX = f'{ARTIFACTS_PATH}wann{os.sep}'
    RUN_CHECKPOINT = f'{RESULTS_PATH}_checkpoint{os.sep}'
    RUN_CHECKPOINT_FN = 'run-checkpoint.pkl'

    NUM_WORKERS = config['NUM_WORKERS']
    WANN_ENV_ID = config['WANN_ENV_ID']

    paths = [ARTIFACTS_PATH, VIS_RESULTS_PATH, SAVE_GIF_PATH, TB_LOG_PATH, WANN_OUT_PREFIX, RUN_CHECKPOINT]
    for p in paths:
        if not os.path.isdir(p):
            os.makedirs(p)

    GAME_CONFIG = config['GAME_CONFIG']
    AGENT_CONFIG = config['AGENT']
    ENV_NAME = GAME_CONFIG.env_name

    games = {
        ENV_NAME: GAME_CONFIG
    }

    wtrain.init_games_config(games)
    gym.envs.register(
        id=WANN_ENV_ID,
        entry_point=config['ENTRY_POINT'],
        max_episode_steps=GAME_CONFIG.max_episode_length
    )

    wann_param_config = config['WANN_PARAM_CONFIG']
    wann_args = dict(
        hyperparam=wann_param_config,
        outPrefix=WANN_OUT_PREFIX,
        num_workers=NUM_WORKERS,
        games=games
    )

    # TODO: re-add vec env
    ENV_ID = WANN_ENV_ID if run_config.USE_WANN else ENV_NAME
    expl_env = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)

    alg_params = AGENT_CONFIG['alg_params']
    if GAME_CONFIG.alg == task.ALG.SAC:
        if run_config.USE_PREV_EXPERIMENT:
            m = alg.load()  # TODO: load SAC model here
        else:
            train_params = AGENT_CONFIG['train_params']
            q_net, v_net, policy_net = alg.vanilla_nets(expl_env, AGENT_CONFIG['n_hidden'],
                                                        AGENT_CONFIG['n_depth'],
                                                        clip_val=AGENT_CONFIG['clip_val'])

            mem = alg.simple_mem(AGENT_CONFIG['mem_size'], expl_env)
            m = alg.SAC(eval_env, expl_env, mem, policy_net,
                        q_net, v_net, train_params, alg_params)
    else:
        raise Exception(f'Algorithm configured is not currently supported')

    # Use MPI if parallel
    if run_config.TRAIN_WANN:
        if "parent" == mpi_fork(NUM_WORKERS+1): os._exit(0)

    for i in range(1, run_config.NUM_TRAIN_STEPS+1):
        if run_config.TRAIN_WANN:
            # TODO: get critic and pass to wann here
            wtrain.run(wann_args, use_checkpoint=True if i > 1 or run_config.START_FROM_LAST_RUN else False,
                       run_train=True)

        if rank == 0:  # if main process
            if i % LOG_INTERVAL == 0:
                print(f'performing learning step {i}/{run_config.NUM_TRAIN_STEPS} complete...')

            # TODO:  SAC learning / logging / checkpointing here
            m.learn()
            m.save(ARTIFACTS_PATH+task.MODEL_ARTIFACT_FILENAME)
            print('TRAINING ALG STEP COMPLETE')  # TODO: add proper logging
        else:
            break  # break if subprocess

        if i % LOG_INTERVAL == 0:
            print(f'step {i}/{run_config.NUM_TRAIN_STEPS} complete')

    if rank == 0:  # if main process
        if run_config.RENDER_TEST_GIFS:
            vid_len = config['VIDEO_LENGTH']

            ENV_ID = WANN_ENV_ID if run_config.USE_WANN else ENV_NAME
            render_agent(m, ENV_ID, vid_len, SAVE_GIF_PATH, filename=f'{run_config.EXPERIMENT_ID}-agent.gif')
            render_agent(m, ENV_ID, vid_len, SAVE_GIF_PATH, filename='random.gif')

    wtrain.run({}, kill_slaves=True)


def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )

    # TODO: check if linux or windows here
    # subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    # ADDED local mod to work with Win 10
    subprocess.check_call(["mpiexec", "-n", str(n), sys.executable] + ['-u'] + sys.argv, env=env)

    return "parent"
  else:
    global nWorker, rank
    nWorker = comm.Get_size()
    rank = comm.Get_rank()
    return "child"


def render_agent(model, env_name, vid_len,
                 out_path, filename, rand_agent=False, render_gif=True):
    if render_gif:
        with gym.make(env_name) as test_env:
            images = []
            obs = test_env.reset()
            for _ in range(vid_len):
                img = test_env.render(mode='rgb_array')
                images.append(img)

                if rand_agent:
                    a = test_env.action_space.sample()
                else:
                    a = model.predict(obs, deterministic=True)[0]

                obs, _, done, _ = test_env.step(a)
                if done:
                    obs = test_env.reset()

                imageio.mimsave(f'{out_path}{filename}',
                                [np.array(img) for i, image in enumerate(images) if i % 2 == 0], fps=30)


# TODO: proper logging
def main():
    if run_config.TASK in ['cartpole-balance']:
        run(cartpole.get_task_config())
    if run_config.TASK in ['bipedal-walker']:
        run(bipedal_walker.get_task_config())
    else:
        raise Exception('No implemented environment found. Please refer to list of implemented environments in README')


if __name__ == '__main__':
    main()
