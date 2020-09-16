class Agent(object):
    def __init__(self, env, eval_env, mem, nets, train_step_params):
        self._env = env
        self._eval_env = eval_env
        self._mem = mem
        self._nets = nets
        self._train_step_params = train_step_params
