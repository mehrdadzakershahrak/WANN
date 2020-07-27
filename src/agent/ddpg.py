import gym
import numpy as np
import torch as T
from component.reflect_mem import ReflectMem
import copy
import torch.optim as optim
import torch.nn.functional as F

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class DdpgAgent(object):
    def __init__(self, actor_model, critic_model, env, actor_lr=1e-3,
                 critic_lr=1e-3, mem_capacity=1000000, learn_batch_size=10000, tau=5e-2, gamma=.9):
        super(DdpgAgent, self).__init__()

        self.env = env
        self.tau = tau
        self.gamma = gamma

        self.mem = ReflectMem(mem_capacity)
        self.mem_capacity = mem_capacity

        actor_model = actor_model.to(device)
        critic_model = critic_model.to(device)

        self.actor = actor_model
        self.actor_target = copy.deepcopy(actor_model)

        self.critic = critic_model
        self.critic_target = copy.deepcopy(critic_model)

        self.action_space = env.action_space
        self.learn_batch_size = learn_batch_size

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.critic_loss = F.smooth_l1_loss

    def perceive(self, env, a):
        return env.step(a)

    def learn(self):
        s, a, r, ns, done = zip(*self.mem.sample(self.learn_batch_size))
        s, a, r, ns, done = T.from_numpy(np.stack(s)).float(), T.from_numpy(np.stack(a)).int(), \
                            T.from_numpy(np.stack(r)).float(), T.from_numpy(np.stack(ns)).float(), \
                            np.stack(done)

        qvals = self.critic(s, a)
        na = self.actor_target(ns)
        nqvals = self.critic_target(ns, na)
        for i, d in enumerate(done):
            if d:
                nqvals[i] = 0.0

        qvals_w_discount = r+self.gamma*nqvals

        critic_loss = self.critic_loss(qvals, qvals_w_discount)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft weight updates to target networks
        for m, m_target in [(self.critic, self.critic_target), (self.actor, self.actor_target)]:
            for target_p, p in zip(m.parameters(), m_target.parameters()):
                target_p.data.copy_((1.0-self.tau)*target_p.data+self.tau*p.data)

    def act(self, s):
        s = s.to(device)
        with T.no_grad():
            acts = self.policy_model(s)
            a = acts.cpu().numpy()  # TODO: add ornstein uhlenbeck random noise for exploration

        return a

    def train(self, init_state,
              episodes,
              max_timesteps,
              use_plot=False):
        env = self.env

        s = init_state

        c = 0
        for ei in range(1, episodes + 1):
            for t in range(1, max_timesteps):
                a = self.act(T.from_numpy(s).float())

                ns, r, done, _ = self.perceive(env, a)

                # TODO: handle done states here
                self.mem.add((s, a, r, ns, done))

                s = ns

                if self.mem.size() < self.mem_capacity:
                    continue

                self.learn()

                if c % 10000 == 0:
                    self.target_model = copy.deepcopy(self.policy_model)
                    c = 0

                c += 1

            # TODO proper logging with DEBUG MODE
            if ei % 1000 == 0:
                print(ei)

        # TODO: remove me when test complete
        print('completed setup')

        # TODO: visualize results here

    def act_optimal(self, s):
        s = s.to(device)
        acts = self.model(s)

        return np.argmax(acts.cpu().numpy()).item()
