import os
import copy
from collections import namedtuple

TASK = 'wann-cartpolebalance-v1'
SEED = 0  # high level seed for all experiments

SEED_RANGE_MIN = 1
SEED_RANGE_MAX = 100000000

RESULTS_PATH = f'result/wann-artifacts/'
ARTIFACTS_PATH = 'model'
TFBOARD_LOG_PATH = 'tf-log'

USE_PREV_EXPERIMENT = False
PREV_EXPERIMENT_PATH = f'{RESULTS_PATH}/wann-ppo2-model'

EXPERIMENT_ID = 'without-wann'

SHOULD_TRAIN_WANN = False
SHOULD_USE_WANN = False

SHOW_TESTS = False
SAVE_VIDEO_PATH = f'result/wann-artifacts/video/'

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
                           'input_size', 'output_size', 'layers', 'i_act', 'h_act',
                           'o_act', 'weightCap', 'noise_bias', 'output_noise',
                           'max_episode_length', 'in_out_labels'])

_default_wann_hyperparams = {
    "task": None,
    "maxGen": 1024,
    "alg_nReps": 3,
    "popSize": 192,
    "select_eliteRatio": 0.2,
    "select_tournSize": 8,
    "alg_wDist": "standard",
    "alg_nVals": 6,
    "alg_probMoo": 0.80,
    "alg_act": 0,
    "alg_speciate": "none",
    "prob_crossover": 0.0,
    "prob_mutAct":  0.50,
    "prob_addNode": 0.25,
    "prob_addConn": 0.20,
    "prob_enable":  0.05,
    "prob_initEnable": 0.5,
    "select_cullRatio": 0.2,
    "save_mod": 8,
    "bestReps": 20
}


def get_default_wann_hyperparams():
    return copy.deepcopy(_default_wann_hyperparams)
