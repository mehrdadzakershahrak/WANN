import gym
from model.dqn_net import DqnVnn
from agent.dqn import DqnAgent


def brickbreak_ram():
    n_traject = 50

    env = gym.make('Breakout-ram-v0').unwrapped

    n_acts = env.action_space.n
    n_obs_space = env.observation_space.shape[0]

    s = env.reset()

    model = DqnVnn(n_obs_space, n_acts)
    agent = DqnAgent(model, env)

    agent.train(init_state=s, episodes=100000, max_timesteps=25)

    print('agent training complete')

    env.close()

