from agent.agent import Agent  # TODO better naming here
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy
import torch
import rlkit.torch.pytorch_util as torch_util
from agent.mem import Mem
import os
import pickle
import config as run_config


if torch.cuda.is_available():
    torch_util.set_gpu_mode(True)


log = run_config.log()


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

        self._train_step_params = train_step_params

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
        if self._mem.num_steps_can_sample() < batch_size:
            return

        for _ in range(n_train_steps):
            batch = self._mem.random_batch(batch_size)
            self._alg.train(batch)

    def learn(self, results_path, seed_mem=True, **kwargs):
        replay_sample_ratio = kwargs['replay_sample_ratio']
        n_episodes = kwargs['n_episodes']
        episode_len = kwargs['episode_len']
        eval_episode_len = kwargs['eval_episode_len']
        start_steps = kwargs['start_steps']
        n_trains_per_step = kwargs['n_trains_per_step']
        eval_interval = kwargs['eval_interval']
        batch_size = kwargs['train_batch_size']
        checkpoint_interval = kwargs['checkpoint_interval']
        log_interval = kwargs['log_interval']
        artifact_path = results_path+f'artifact'

        # TODO: log summary of current config

        if seed_mem:
            s = self._env.reset()
            for k in range(start_steps):
                a = self._env.action_space.sample()
                ns, r, done, _ = self._env.step(a)

                self._mem.add_sample(observation=s, action=a, reward=r, next_observation=ns,
                                     terminal=1 if done else 0, env_info=dict())
                if done:
                    s = self._env.reset()
                else:
                    s = ns

        # TODO: use RAY Sampler for parallel simulation sampling
        train_rt = Agent.results_tracker(id='train_performance')

        s = self._env.reset()
        # TODO: track and log policy loss
        for _ in range(n_episodes):
            for i in range(1, episode_len+1):
                a = self.pred(s)
                ns, r, done, _ = self._env.step(a)

                self.life_tracker['total_n_train_steps'] += 1
                train_rt['train_interval_timesteps'] += 1

                train_rt['train_rewards'].append(r)
                self._mem.add_sample(observation=s, action=a, reward=r, next_observation=ns,
                                     terminal=1 if done else 0, env_info=dict())

                if i % replay_sample_ratio == 0:
                    self._train_step(n_trains_per_step, batch_size)
                self.life_tracker['total_n_train_batches'] += 1

                if i % checkpoint_interval == 0:
                    self.save(artifact_path)

                if done or i == episode_len:
                    s = self._env.reset()

                    self.life_tracker['total_n_train_episodes'] += 1
                    train_rt['n_train_episodes_since_last_log'] += 1

                    if self.life_tracker['total_n_train_episodes'] % log_interval == 0:
                        if self.life_tracker['total_n_train_episodes'] % eval_interval == 0:
                            self.life_tracker['total_n_evals'] += 1
                            train_rt['eval_interval_timesteps'] = eval_episode_len

                            s = self._eval_env.reset()
                            for _ in range(eval_episode_len):
                                a = self.pred(s)

                                ns, r, done, _ = self._eval_env.step(a)
                                train_rt['eval_rewards'].append(r)

                                if done:
                                    s = self._eval_env.reset()
                                else:
                                    s = ns

                        self.log_performance(train_rt)
                        train_rt = Agent.results_tracker(id='train_performance')
                else:
                    s = ns

    def pred(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(torch_util.device)
        with torch.no_grad():
            return self._policy_net(state, deterministic=deterministic)[0].cpu().detach().numpy()

    def save(self, filepath):
        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        nets = [self._policy_net, self._q1_net, self._q2_net,
                self._target_q1_net, self._target_q2_net]
        net_fps = ['policy-net.pt', 'q1-net.pt', 'q2-net.pt',
                   'target-q1-net.pt', 'target-q2-net.pt']
        for i, fn in enumerate(net_fps):
            torch.save(nets[i], f'{filepath}{os.sep}{fn}')

        comps = [self._train_step_params]
        comp_fps = ['train-step-params.pkl']
        for i, fn in enumerate(comp_fps):
            with open(f'{filepath}{os.sep}{fn}', 'wb') as f:
                pickle.dump(comps[i], f)


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
        action_dim=act_size
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

    nets = [q1_net, q2_net, policy_net]
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


def load(env, eval_env, mem, filepath):
    policy_net = torch.load(f'{filepath}policy-net.pt')
    q1_net = torch.load(f'{filepath}q1-net.pt')
    q2_net = torch.load(f'{filepath}q2-net.pt')
    target_q1_net = torch.load(f'{filepath}target-q1-net.pt')
    target_q2_net = torch.load(f'{filepath}target-q2-net.pt')

    nets = dict(
        policy_net=policy_net,
        q1_net=q1_net,
        q2_net=q2_net,
        target_q1_net=target_q1_net,
        target_q2_net=target_q2_net
    )

    with open(f'{filepath}{os.sep}train-step-params.pkl', 'rb') as f:
        train_step_params = pickle.load(f)

    return SAC(env, eval_env, mem, nets, train_step_params)


def simple_mem(size, env):
    return Mem(size, env)
