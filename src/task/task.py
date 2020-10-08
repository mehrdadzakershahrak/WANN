import enum
import copy
from collections import namedtuple
import gym
import os


class ALG(enum.Enum):
    SAC = 1


Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
                           'input_size', 'output_size', 'layers', 'i_act', 'h_act',
                           'o_act', 'weightCap', 'noise_bias', 'output_noise',
                           'max_episode_length', 'n_critic_bootstrap',
                           'alg_type', 'artifacts_path', 'in_out_labels'])

_DEFAULT_WANN_HYPERPARAMS = {
    "task": None,
    "maxGen": 100,
    "alg_nReps": 3,
    "popSize": 192,
    "select_eliteRatio": 0.2,
    "select_tournSize": 16,
    "alg_wDist": "standard",
    "alg_nVals": 6,
    "alg_probMoo": 0.80,
    "alg_act": 0,
    "alg_speciate": "none",
    "prob_crossover": 0.0,
    "prob_mutAct": 0.50,
    "prob_addNode": 0.25,
    "prob_addConn": 0.20,
    "prob_enable": 0.05,
    "prob_initEnable": 0.25,
    "select_cullRatio": 0.2,
    "save_mod": 8,
    "bestReps": 20,
    "spec_target": 4
}

MODEL_ARTIFACT_FILENAME = 'primary-model'
RESULTS_PATH = f'result{os.sep}'


class ObsDatPrepWrapper(gym.ObservationWrapper):
    def __init__(self, env, datprep):
        super().__init__(env)

        self.datprep = datprep

    def observation(self, obs):
        obs = self.datprep(obs)

        return obs


def make_env(env_name, datprep=None):
    return ObsDatPrepWrapper(gym.make(env_name), datprep) \
        if datprep is not None else gym.make(env_name)


def get_default_wann_hyperparams():
    return copy.deepcopy(_DEFAULT_WANN_HYPERPARAMS)
