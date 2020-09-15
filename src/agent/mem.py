from rlkit.data_management.env_replay_buffer import EnvReplayBuffer


class Mem(EnvReplayBuffer):
    def __init__(self, size, env):
        super().__init__(
            env=env,
            max_replay_buffer_size=size
        )

    def add_sample(self, observation, action, reward, next_observation, terminal,
                   **kwargs):
        return super().add_sample(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            terminal=terminal,
            **kwargs
        )
