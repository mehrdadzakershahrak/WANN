# base agent for common logging
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit


class Agent(object):
    def __init__(self, env, eval_env, mem, train_params, alg_params, nets):
        self._env = eval_env
        self._eval_env = eval_env
        self._mem = mem
        self._train_params = train_params
        self._alg_params = alg_params
        self._nets = nets
