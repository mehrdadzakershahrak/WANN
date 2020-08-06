TASK = 'wann-cartpolebalance-v1'
SEED = 0  # high level seed for all experiments

RESULTS_PATH = 'result'
ARTIFACTS_PATH = 'model'
TFBOARD_LOG_PATH = 'tf-log'

USE_PREV_EXPERIMENT = False
PREV_EXPERIMENT_PATH = None

# parser.add_argument('-d', '--default', type=str, \
#                     help='default hyperparameter file', default='extern/wann/p/default_wann.json')
# parser.add_argument('-p', '--hyperparam', type=str, \
#                     help='hyperparameter file', default='extern/wann/p/laptop_swing.json')
# parser.add_argument('-o', '--outPrefix', type=str, \
#                     help='file name for result output', default='test')
# parser.add_argument('-n', '--num_worker', type=int, \
#                     help='number of cores to use', default=mp.cpu_count())
