from agent import Agent
from copy import copy
from rlkit.torch.sac.sac import SACTrainer as SoftActorCritic_rlkit


class SAC(Agent):
    def __init__(self, env, replay, policy_net, q_net, v_net, params):
        super().__init__(env, replay, nets=(policy_net, q_net, v_net, params))

        self._params = params
        self._env = env
        self._replay = replay
        self._policy_net = policy_net
        self._q_net = q_net
        self._v_net = v_net

        self._target_q_net = copy.deepcopy(q_net)
        self._target_v_net = copy.deepcopy(v_net)

        self._alg = SoftActorCritic_rlkit(
            env=self._env,
            policy=self._policy_net,
            qf1=self._v_net,
            qf2=self._q_net,
            target_qf1=self._target_v_net,
            target_qf2=self._target_q_net,
            use_automatic_entropy_tuning=False,
            **self._params
        )

    def train(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def pred(self, deterministic=True):
        pass
