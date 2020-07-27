import gym
from model.dqn_net import DqnVnn
from agent.ddpg import DdpgAgent


def half_cheetah_v2():
    env = gym.make('HalfCheetah-v2').unwrapped

    n_acts = env.action_space.n
    n_obs_space = env.observation_space.shape[0]

    s = env.reset()

    actor_model = DqnVnn(n_obs_space, n_acts)
    critic_model = DqnVnn(n_obs_space, n_acts)

    agent = DdpgAgent(actor_model, critic_model, env)

    agent.train(init_state=s, episodes=100000, max_timesteps=25)

    # TODO: add score visualization

    print('agent training complete')

    env.close()
