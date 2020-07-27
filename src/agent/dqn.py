import gym
import numpy as np
import torch as T
from component.reflect_mem import ReflectMem
import copy
import torch.optim as optim
import torch.nn.functional as F

device = T.device('cuda' if T.cuda.is_available() else 'cpu')


class DqnAgent(object):
    def __init__(self, model, env, lr=1e-3, mem_capacity=1000000,
                 learn_batch_size=10000, gamma=.9):
        super(DqnAgent, self).__init__()

        self.env = env

        self.gamma = gamma

        self.mem = ReflectMem(mem_capacity)
        self.mem_capacity = mem_capacity

        model = model.to(device)

        self.policy_model = model
        self.target_model = copy.deepcopy(model)

        self.action_space = env.action_space
        self.learn_batch_size = learn_batch_size

        self.optimizer = optim.Adam(self.policy_model.parameters(), lr=lr)
        self.loss = F.smooth_l1_loss

    def perceive(self, env, a):
        return env.step(a)

    def learn(self):
        s, a, r, ns, done = zip(*self.mem.sample(self.learn_batch_size))
        s, a, r, ns, done = T.from_numpy(np.stack(s)).float(), T.from_numpy(np.stack(a)).int(), \
                            T.from_numpy(np.stack(r)).float(), T.from_numpy(np.stack(ns)).float(), \
                            np.stack(done)

        ns_vals = T.max(self.target_model(s), dim=1).values.to(device)
        for i, d in enumerate(done):
            if d:
                ns_vals[i] = 0.0

        q_target = r+(ns_vals*self.gamma)
        q_est = T.max(self.policy_model(s), dim=1).values.to(device)

        self.optimizer.zero_grad()
        self.loss(q_est, q_target).backward()

        # TODO: add gradient clipping
        # for p in self.policy_model.parameters():
        #     print('test')
        #     p.grad.data.clamp(-1, 1)

        self.optimizer.step()

    def act(self, s, epsilon):
        s = s.to(device)
        sample = np.random.random_sample()

        if sample <= 1 - epsilon:
            with T.no_grad():
                acts = self.policy_model(s)

                a = np.argmax(acts.cpu().numpy()).item()
        else:
            a = self.action_space.sample()

        return a

    def train(self, init_state,
              episodes,
              max_timesteps,
              init_ep=1.0,
              ep_decay=.995, ep_end=.01,
              use_plot=False):
        ep = init_ep
        env = self.env

        s = init_state

        c = 0
        for ei in range(1, episodes + 1):
            for t in range(1, max_timesteps):
                a = self.act(T.from_numpy(s).float(), ep)

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

            ep = max(ep_decay*ep, ep_end)

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
