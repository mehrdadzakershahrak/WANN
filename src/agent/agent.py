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
            n_episode_steps=int(0),
            n_evals=int(0)
        )

        self._mem = mem

    def updt_episode_cnts(self, results_tracker):
        rewards = np.array(results_tracker['rewards'])

        results_tracker['episode_mean_rewards'] = np.mean(rewards)/float(results_tracker['n_episodes'])
        results_tracker['episode_min_rewards'] = np.min(rewards)/float(results_tracker['n_episodes'])
        results_tracker['episode_max_rewards'] = np.max(rewards)/float(results_tracker['n_episodes'])

    def log_performance(self, results_tracker):
        summary = dict(
            cur_mem_size=self._mem.num_steps_can_sample(),
            episode_mean_rewards=results_tracker['episode_mean_rewards'],
            episode_min_rewards=results_tracker['episode_min_rewards'],
            episode_max_rewards=results_tracker['episode_max_rewards'],
        )
        summary.update(self.life_tracker)
        log.info(summary)

    @staticmethod
    def results_tracker(id):
        return copy.deepcopy(dict(
            id=id,
            rewards=[],
            episode_mean_rewards=0.0,
            episode_max_rewards=0.0,
            episode_min_rewards=0.0,
            n_episodes=int(0)
        ))
