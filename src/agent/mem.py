from rlkit.data_management.simple_replay_buffer import SimpleReplayBuffer


class Mem(SimpleReplayBuffer):
    def __init__(self, size, env):
        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.shape[0]

        super().__init__(
            max_replay_buffer_size=size,
            observation_dim=obs_size,
            action_dim=act_size,
            env_info_sizes=dict()
        )

        self._buf_cap = size

    def add_sample(self, observation, action, reward, next_observation, terminal,
                   **kwargs):
        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=1 if terminal.item() else 0,
            **kwargs
        )
