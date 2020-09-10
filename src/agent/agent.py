# base agent for common logging
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit


class Agent(object):
    def __init__(self, conf, env, replay, nets):
        self._conf = conf
        self._env = env
        self._replay = replay
        self._nets = nets
