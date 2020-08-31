import enum
import copy
from collections import namedtuple
import config
import gym
from extern.wann.neat_src import ann as wnet
import os


class ALG(enum.Enum):
    PPO = 1
    DDPG = 2
    TD3 = 3


Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
                           'input_size', 'output_size', 'layers', 'i_act', 'h_act',
                           'o_act', 'weightCap', 'noise_bias', 'output_noise',
                           'max_episode_length', 'alg', 'artifacts_path', 'in_out_labels'])

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

RESULTS_PATH = f'result{os.sep}'
MODEL_ARTIFACT_FILENAME = 'primary-model'


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, champion_artifacts_path):
        super().__init__(env)

        self.wVec, self.aVec, _ = wnet.importNet(champion_artifacts_path)

    def observation(self, obs):
        if config.USE_WANN:
            obs = wnet.act(self.wVec, self.aVec,
                           nInput=obs.shape[0],
                           nOutput=obs.shape[0],
                           inPattern=obs)

        return obs


def get_default_wann_hyperparams():
    return copy.deepcopy(_DEFAULT_WANN_HYPERPARAMS)
