from agent.agent import Agent  # TODO better naming here
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
import torch
import rlkit.torch.pytorch_util as torch_util
from agent.mem import Mem

if torch.cuda.is_available():
    torch_util.set_gpu_mode(True)


class SAC(Agent):
    def __init__(self, env, eval_env, mem, nets,
                 train_step_params):
        super().__init__(env, eval_env, mem, nets, train_step_params)
        self._mem = mem

        self._env = env
        self._eval_env = eval_env

        self._policy_net, self._q1_net, self._q2_net, self._target_q1_net,\
        self._target_q2_net = nets['policy_net'], nets['q1_net'], nets['q2_net'],\
                              nets['target_q1_net'], nets['target_q2_net']

        self._eval_policy_net = MakeDeterministic(self._policy_net)

        self._alg = SACTrainer(
            env=self._env,
            policy=self._policy_net,
            qf1=self._q1_net,
            qf2=self._q2_net,
            target_qf1=self._target_q1_net,
            target_qf2=self._target_q2_net,
            **train_step_params
        )

    def _train_step(self, n_train_steps, batch_size):
        for _ in range(n_train_steps):
            batch = self._mem.random_batch(batch_size)
            self._alg.train(batch)

    def learn(self, **kwargs):
        episode_len = kwargs['episode_len']
        eval_episode_len = kwargs['eval_episode_len']
        start_steps = kwargs['start_steps']
        n_train_steps = kwargs['n_train_steps']
        train_epochs = kwargs['train_epochs']
        eval_interval = kwargs['eval_interval']
        batch_size = kwargs['batch_size']

        for i in range(train_epochs):
            s = self._env.reset()

            # TODO: rewards and returns tracking
            # TODO: get policy loss

            # TODO: DRY ME UP
            steps = [start_steps, episode_len]
            for k, stp in enumerate(steps):
                for _ in range(stp):
                    if k == 0:
                        a = self._env.action_space.sample()
                    else:
                        a = self.pred(s)

                    ns, r, done, _ = self._env.step(a)

                    self._mem.add_sample(observation=s, action=a, reward=r, next_observation=ns,
                                         terminal=1 if done else 0, env_info=dict())
                    if done:
                        s = self._env.reset()
                    else:
                        s = ns

            self._train_step(n_train_steps, batch_size)

            if i % eval_interval == 0:
                s = self._eval_env.reset()
                eval_rewards = []
                eval_G = []  # TODO: backed up returns
                for _ in range(eval_episode_len):
                    a = self.pred(s)

                    ns, r, done, _ = self._eval_env.step(a)

                    eval_rewards.append(r)
                    if done:
                        s = self._eval_env.reset()
                    else:
                        s = ns

                # TODO: log eval here

    # TODO: change eval pred to deterministic
    def pred(self, state):
        state = torch.from_numpy(state).float().to(torch_util.device)
        with torch.no_grad():
            return self._policy_net(state)[0].cpu().detach().numpy()

    def save(self, filepath):
        pass


def vanilla_nets(env, n_lay_nodes, n_depth, clip_val=1):
    hidden = [n_lay_nodes]*n_depth

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]

    q1_net = FlattenMlp(
        hidden_sizes=hidden,
        input_size=obs_size+act_size,
        output_size=1,
    ).to(device=torch_util.device)

    q2_net = FlattenMlp(
        hidden_sizes=hidden,
        input_size=obs_size+act_size,
        output_size=1,
    ).to(device=torch_util.device)

    policy_net = TanhGaussianPolicy(
        hidden_sizes=hidden,
        obs_dim=obs_size,
        action_dim=act_size,
    ).to(device=torch_util.device)

    target_q1_net = FlattenMlp(
        hidden_sizes=hidden,
        input_size=obs_size + act_size,
        output_size=1,
    ).to(device=torch_util.device)

    target_q2_net = FlattenMlp(
        hidden_sizes=hidden,
        input_size=obs_size + act_size,
        output_size=1,
    ).to(device=torch_util.device)

    nets = [q1_net, q2_net, policy_net, target_q1_net, target_q2_net]
    for n in nets:
        for p in n.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_val, clip_val))

    return dict(
        policy_net=policy_net,
        q1_net=q1_net,
        q2_net=q2_net,
        target_q1_net=target_q1_net,
        target_q2_net=target_q2_net
    )


def load(filepath):
    pass


def simple_mem(size, env):
    return Mem(size, env)
