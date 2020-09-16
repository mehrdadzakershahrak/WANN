import config as run_config
import copy
import numpy as np

log = run_config.log()


class Agent(object):
    def __init__(self, env, eval_env, mem, nets, train_step_params):
        # TODO: add common interface for agents in future iteration
        self.life_tracker = dict(
            n_train_epochs=int(1),
            n_train_batches=int(0),
            n_train_episodes=int(0),
            n_train_steps=int(0),
            n_train_episode_steps=int(0),
            n_evals=int(0)
        )

        self._mem = mem

    def log_performance(self, results_tracker):
        rewards = np.array(results_tracker['rewards'])
        n_episodes_since_last_log = results_tracker['n_episodes_since_last_log']
        summary = dict(
            cur_mem_size=self._mem.num_steps_can_sample(),
            episode_mean_rewards=rewards.mean(),
            episode_min_rewards=rewards.min(),
            episode_max_rewards=rewards.max(),
            n_episodes_since_last_log=n_episodes_since_last_log
        )
        summary.update(self.life_tracker)
        log.info(summary)

    @staticmethod
    def results_tracker(id):
        return copy.deepcopy(dict(
            id=id,
            rewards=[],
            n_episodes_since_last_log=int(0)
        ))
