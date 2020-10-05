from task import lunar_lander, bipedal_walker, ant, bipedal_walker_hardcore, car_racing, \
    cartpole, half_cheetah, humanoid, lunar_lander
from extern.wann import wann_train as wtrain
from stable_baselines3 import SAC
import gym
from task import task
import imageio
import numpy as np
import config as run_config
import sys
from mpi4py import MPI
import subprocess
import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback

comm = MPI.COMM_WORLD
rank = 0

SEED_RANGE_MIN = 1
SEED_RANGE_MAX = 100000000
LOG_INTERVAL = 10

log = run_config.log()


def run(config):
    log.info(f'Beginning run for experiment {run_config.EXPERIMENT_ID}')

    RESULTS_PATH = config['RESULTS_PATH']
    EXPERIMENTS_PREFIX = f'{RESULTS_PATH}{run_config.EXPERIMENT_ID}{os.sep}'
    ARTIFACTS_PATH = f'{EXPERIMENTS_PREFIX}artifact{os.sep}'
    VIS_RESULTS_PATH = f'{EXPERIMENTS_PREFIX}vis{os.sep}'
    SAVE_GIF_PATH = f'{EXPERIMENTS_PREFIX}gif{os.sep}'
    WANN_OUT_PREFIX = f'{ARTIFACTS_PATH}wann{os.sep}'
    ALG_OUT_PREFIX = f'{ARTIFACTS_PATH}alg{os.sep}'
    NUM_WORKERS = config['NUM_WORKERS']
    WANN_ENV_ID = config['WANN_ENV_ID']
    GAME_CONFIG = config['GAME_CONFIG']
    AGENT_CONFIG = config['AGENT']

    log.info('RUN CONFIG:')
    log.info(config)

    log.info('Experiment description:')
    log.info(run_config.DESCRIPTION)

    paths = [ARTIFACTS_PATH, VIS_RESULTS_PATH, SAVE_GIF_PATH, WANN_OUT_PREFIX,
             f'{ALG_OUT_PREFIX}checkpoint{os.sep}checkpoint-alg{os.sep}']
    for p in paths:
        if not os.path.isdir(p):
            os.makedirs(p)

    ENV_NAME = GAME_CONFIG.env_name

    games = {
        ENV_NAME: GAME_CONFIG
    }

    wtrain.init_games_config(games)

    if run_config.TRAIN_WANN:
        if "parent" == mpi_fork(NUM_WORKERS + 1): os._exit(0)

    wann_param_config = config['WANN_PARAM_CONFIG']
    wann_args = dict(
        hyperparam=wann_param_config,
        outPrefix=WANN_OUT_PREFIX,
        rank=rank,
        num_workers=NUM_WORKERS,
        games=games
    )

    alg = None
    for i in range(1, run_config.NUM_TRAIN_STEPS + 1):
        if run_config.TRAIN_WANN:
            wtrain.run(wann_args, use_checkpoint=True if i > 1 or run_config.USE_PREV_EXPERIMENT else False,
                       alg_critic=None if alg is None else alg.critic,
                       mem=None if alg is None else alg.replay_buffer)

        if rank == 0:  # main proc
            if i <= 1:
                gym.envs.register(
                    id=WANN_ENV_ID,
                    entry_point=config['ENTRY_POINT'],
                    max_episode_steps=GAME_CONFIG.max_episode_length
                )

                learn_params = AGENT_CONFIG['learn_params']
                ENV_ID = WANN_ENV_ID if run_config.USE_WANN else ENV_NAME
                env = Monitor(gym.make(ENV_ID), f'{EXPERIMENTS_PREFIX}log')
                checkpoint_callback = CheckpointCallback(save_freq=learn_params['alg_checkpoint_interval'],
                                                         save_path=f'{ALG_OUT_PREFIX}checkpoint{os.sep}checkpoint-alg')
                eval_env = gym.make(ENV_ID)
                eval_callback = EvalCallback(eval_env,
                                             best_model_save_path=f'{ALG_OUT_PREFIX}checkpoint{os.sep}eval-best-alg',
                                             log_path=f'{EXPERIMENTS_PREFIX}log{os.sep}checkpoint',
                                             eval_freq=learn_params['eval_interval'])
                cb = CallbackList([checkpoint_callback, eval_callback])

                learn_params = AGENT_CONFIG['learn_params']
                # TODO: save/load if on wann or SAC optimize step for prev experiment starts
                if GAME_CONFIG.alg_type == task.ALG.SAC:
                    if run_config.USE_PREV_EXPERIMENT:
                        alg = SAC.load(f'{run_config.PREV_EXPERIMENT_PATH}{os.sep}alg')  # TODO: load SAC model here
                    else:
                        alg = SAC(AGENT_CONFIG['policy'], env, verbose=learn_params['log_verbose'],
                                  tensorboard_log=f'{EXPERIMENTS_PREFIX}log{os.sep}tb-log',
                                  buffer_size=learn_params['mem_size'], learning_rate=learn_params['learn_rate'],
                                  learning_starts=learn_params['start_steps'],
                                  batch_size=learn_params['train_batch_size'],
                                  tau=learn_params['tau'], gamma=learn_params['gamma'],
                                  train_freq=learn_params['n_trains_per_step'],
                                  target_update_interval=learn_params['replay_sample_ratio'],
                                  gradient_steps=learn_params['gradient_steps_per_step'],
                                  n_episodes_rollout=learn_params['episode_len'],
                                  target_entropy=learn_params['target_entropy'],
                                  device=config['DEVICE'])
                else:
                    raise Exception(f'Algorithm configured is not currently supported')

            if i > 1:
                alg.learning_starts = 0

            if i % LOG_INTERVAL == 0:
                log.info(f'performing learning step {i}/{run_config.NUM_TRAIN_STEPS} complete...')
            log.info('PERFORMING ALG TRAIN STEP')
            alg.learn(total_timesteps=learn_params['timesteps'], log_interval=learn_params['log_interval'],
                      callback=cb)
            alg.save(f'{ALG_OUT_PREFIX}checkpoint{os.sep}full-run-checkpoint{os.sep}checkpoint-step-{i}')
        else:
            break  # break if subprocess

        if i % LOG_INTERVAL == 0:
            log.info(f'step {i}/{run_config.NUM_TRAIN_STEPS} complete')

    if rank == 0:  # if main process
        if run_config.RENDER_TEST_GIFS:
            vid_len = config['VIDEO_LENGTH']

            ENV_ID = WANN_ENV_ID if run_config.USE_WANN else ENV_NAME
            render_agent(alg, ENV_ID, vid_len, SAVE_GIF_PATH, filename=f'{run_config.EXPERIMENT_ID}-agent.gif')
            render_agent(alg, ENV_ID, vid_len, SAVE_GIF_PATH, filename='random.gif')

    wtrain.run(None, kill_slaves=True)


def mpi_fork(n):
    if n <= 1:
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


def main():
    if run_config.TASK in ['cartpole']:
        run(cartpole.get_task_config())
    if run_config.TASK in ['lunar-lander']:
        run(lunar_lander.get_task_config())
    if run_config.TASK in ['bipedal-walker']:
        run(bipedal_walker.get_task_config())
    if run_config.TASK in ['car-racing']:
        run(car_racing.get_task_config())
    if run_config.TASK in ['half-cheetah']:
        run(half_cheetah.get_task_config())
    if run_config.TASK in ['ant']:
        run(ant.get_task_config())
    if run_config.TASK in ['humanoid']:
        run(humanoid.get_task_config())
    else:
        raise Exception('No implemented environment found. Please refer to list of implemented environments in README')


if __name__ == '__main__':
    main()
