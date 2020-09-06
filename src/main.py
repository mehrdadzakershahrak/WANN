from task import cartpole, bipedal_walker
from extern.wann import wann_train as wtrain
from extern.wann.neat_src import ann as wnet
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, DDPG
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import MlpPolicy as DDPG_MlpPolicy
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
import tensorflow as tf
import pickle
from stable_baselines.common.noise import NormalActionNoise, AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

tf.get_logger().setLevel('FATAL')

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

    if run_config.USE_PREV_EXPERIMENT:
        if GAME_CONFIG.alg == task.ALG.PPO:
            m = PPO2.load(run_config.PREV_EXPERIMENT_PATH)
        elif GAME_CONFIG.alg == task.ALG.DDPG:
            m = DDPG.load(run_config.PREV_EXPERIMENT_PATH)
        else:
            raise('Algorithm chosen is not supported.')
    else:
        if not run_config.START_FROM_LAST_RUN:
            # Take one step first without WANN to ensure primary algorithm model artifacts are stored
            onestep_env = make_vec_env(ENV_NAME, n_envs=1)

            if GAME_CONFIG.alg == task.ALG.PPO:
                m = PPO2(MlpPolicy, onestep_env, verbose=0)
            elif GAME_CONFIG.alg == task.ALG.DDPG:
                n_actions = onestep_env.action_space.shape[-1]
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                            sigma=float(0.5)*np.ones(n_actions))
                m = DDPG(DDPG_MlpPolicy, onestep_env,
                         gamma=AGENT_CONFIG['gamma'],
                         verbose=0,
                         batch_size=AGENT_CONFIG['batch_size'],
                         buffer_size=AGENT_CONFIG['buffer_size'],
                         param_noise=None, action_noise=action_noise)
            else:
                raise ('Algorithm chosen is not supported.')

            m.learn(total_timesteps=1, reset_num_timesteps=False, tb_log_name='__primary-model')
            m.save(ARTIFACTS_PATH + task.MODEL_ARTIFACT_FILENAME)

        # Use MPI if parallel
        if run_config.TRAIN_WANN:
            if "parent" == mpi_fork(NUM_WORKERS +1): os._exit(0)

        if run_config.START_FROM_LAST_RUN:
            with open(RUN_CHECKPOINT + RUN_CHECKPOINT_FN, 'rb') as f:
                run_track = pickle.load(f)
        else:
            run_track = dict(
                wann_step=True,
                alg_step=False,
                total_steps=0
            )

        total_steps = run_track['total_steps']
        for i in range(1, run_config.NUM_TRAIN_STEPS+1):
            total_steps += 1

            if rank == 0 and i % LOG_INTERVAL == 0:
                print(f'performing learning step {i}/{run_config.NUM_TRAIN_STEPS} complete...')

            if run_config.TRAIN_WANN:
                agent_params = m.get_parameters()
                agent_params = dict((key, value) for key, value in agent_params.items())
                wann_args['agent_params'] = agent_params
                wann_args['agent_env'] = m.get_env()
                wann_args['rank'] = rank
                wann_args['nWorker'] = nWorker
                wtrain.run(wann_args, use_checkpoint=True if i > 1 or run_config.START_FROM_LAST_RUN else False,
                           run_train=run_track['wann_step'])

            if not run_config.USE_PREV_EXPERIMENT:
                run_track = dict(
                    wann_step=False,
                    alg_step=True,
                    total_steps=total_steps
                )
                with open(RUN_CHECKPOINT + RUN_CHECKPOINT_FN, 'wb') as f:
                    pickle.dump(run_track, f, protocol=pickle.HIGHEST_PROTOCOL)

            if rank == 0:  # if main process
                # TODO: add callback for visualize WANN interval as well as
                # gif sampling at different stages
                if run_config.VISUALIZE_WANN:
                    champion_path = f'{WANN_OUT_PREFIX}_best.out'
                    wVec, aVec, _ = wnet.importNet(champion_path)

                    wann_vis.viewInd(champion_path, GAME_CONFIG)
                    plt.savefig(f'{VIS_RESULTS_PATH}wann-net-graph.png')

                if i == 1:
                    # TODO: re-add vec env
                    # env = make_vec_env(ENV_ID, n_envs=mp.cpu_count())
                    ENV_ID = WANN_ENV_ID if run_config.USE_WANN else ENV_NAME
                    env = make_vec_env(ENV_ID, n_envs=NUM_WORKERS)

                    if GAME_CONFIG.alg == task.ALG.PPO:
                        if run_config.START_FROM_LAST_RUN:
                            m = PPO2.load(ARTIFACTS_PATH + task.MODEL_ARTIFACT_FILENAME)
                        else:
                            # TODO: configuration for hyperparameters
                            m = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=TB_LOG_PATH)
                    elif GAME_CONFIG.alg == task.ALG.DDPG:
                        if run_config.START_FROM_LAST_RUN:
                            m = DDPG.load(ARTIFACTS_PATH+task.MODEL_ARTIFACT_FILENAME)
                        else:
                            n_actions = env.action_space.shape[-1]
                            action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                        sigma=float(0.5)*np.ones(n_actions))
                            m = DDPG(DDPG_MlpPolicy, onestep_env,
                                     gamma=AGENT_CONFIG['gamma'],
                                     verbose=0,
                                     batch_size=AGENT_CONFIG['batch_size'],
                                     buffer_size=AGENT_CONFIG['buffer_size'],
                                     param_noise=None, action_noise=action_noise,
                                     tensorboard_log=TB_LOG_PATH, full_tensorboard_log=True)
                    elif GAME_CONFIG.alg == task.ALG.TD3:
                        pass
                    else:
                        raise Exception(f'Algorithm configured is not currently supported')

                if run_track['alg_step']:
                    print('TRAINING ALG STEP...')
                    print(AGENT_CONFIG['total_timesteps'])
                    m.learn(total_timesteps=AGENT_CONFIG['total_timesteps'], log_interval=AGENT_CONFIG['log_interval'],
                            reset_num_timesteps=True, tb_log_name='primary-model')
                    m.save(ARTIFACTS_PATH+task.MODEL_ARTIFACT_FILENAME)
                    print('TRAINING ALG STEP COMPLETE')

                    if not run_config.USE_PREV_EXPERIMENT:
                        run_track = dict(
                            wann_step=True,
                            alg_step=False,
                            total_steps=total_steps
                        )
                        with open(RUN_CHECKPOINT + RUN_CHECKPOINT_FN, 'wb') as f:
                            pickle.dump(run_track, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                break  # break if subprocess

            if rank == 0 and i % 10 == 0:
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
