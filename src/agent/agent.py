# base agent for common logging
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit


class Agent(object):
    def __init__(self, eval_env, expl_env, mem, train_params, alg_params, nets):
        self._eval_env = eval_env
        self._expl_env = expl_env
        self._mem = mem
        self._nets = nets
        self._train_params = train_params
        self._alg_params = alg_params
