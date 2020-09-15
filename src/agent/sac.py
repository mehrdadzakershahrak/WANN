from agent.agent import Agent # TODO better naming here
import copy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import ConcatMlp
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import numpy as np
import torch
import rlkit.torch.pytorch_util as torch_util
from agent.mem import Mem

if torch.cuda.is_available():
    torch_util.set_gpu_mode(True)


class SAC(Agent):
    def __init__(self, eval_env, expl_env, mem, policy_net,
                 q1_net, q2_net, target_q1_net, target_q2_net,
                 train_params, alg_params):
        super().__init__(eval_env, expl_env, mem, train_params, alg_params,
                         nets=(policy_net, q1_net, q2_net, target_q1_net,
                               target_q2_net, train_params, alg_params))
        self._mem = mem
        self._policy_net = policy_net
        self._q1_net = q1_net
        self._q2_net = q2_net

        self._target_q1_net = target_q1_net
        self._target_q2_net = target_q2_net

        self._eval_policy_net = MakeDeterministic(policy_net)

        eval_path_collector = MdpPathCollector(
            eval_env,
            self._eval_policy_net
        )
        expl_path_collector = MdpPathCollector(
            expl_env,
            self._policy_net
        )

        self._trainer = SACTrainer(
            env=self._expl_env,
            policy=self._policy_net,
            qf1=self._q1_net,
            qf2=self._q2_net,
            target_qf1=self._target_q1_net,
            target_qf2=self._target_q2_net,
            **train_params
        )
        self._alg = TorchBatchRLAlgorithm(
            trainer=self._trainer,
            exploration_env=self._expl_env,
            evaluation_env=self._eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=mem,
            **alg_params
        )

    def learn(self):
        self._alg.to(torch_util.device)
        self._alg.train()

    def pred(self, state):
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def load(self, filepath):
        pass

    def save(self, filepath):
        pass


def vanilla_nets(env, n_lay_nodes, n_depth, clip_val=1):
    hidden = [n_lay_nodes]*n_depth

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    q1_net = ConcatMlp(
        hidden_sizes=hidden,
        input_size=obs_size+act_size,
        output_size=1,
    ).to(device=torch_util.device)

    q2_net = ConcatMlp(
        hidden_sizes=hidden,
        input_size=obs_size+act_size,
        output_size=1,
    ).to(device=torch_util.device)

    policy_net = TanhGaussianPolicy(
        hidden_sizes=hidden,
        obs_dim=obs_size,
        action_dim=act_size,
    ).to(device=torch_util.device)

    target_q1_net = ConcatMlp(
        hidden_sizes=hidden,
        input_size=obs_size + act_size,
        output_size=1,
    ).to(device=torch_util.device)

    target_q2_net = ConcatMlp(
        hidden_sizes=hidden,
        input_size=obs_size + act_size,
        output_size=1,
    ).to(device=torch_util.device)

    nets = [q1_net, q2_net, policy_net, target_q1_net, target_q2_net]
    for n in nets:
        for p in n.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_val, clip_val))

    return (n for n in nets)


def simple_mem(size, env):
    return Mem(size, env)
