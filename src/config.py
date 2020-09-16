import os

VERSION_NUM = 1000

# GLOBAL CONFIGURABLE PARAMETERS
# DEFAULT CONFIGURATION
############################################
TASK = 'bipedal-walker'
EXPERIMENT_ID = f'wann-ppo-bipedalwalker-v3-{VERSION_NUM}'
SEED = 0  # high level seed for all experiments
USE_PREV_EXPERIMENT = False
PREV_EXPERIMENT_PATH = f'result{os.sep}artifact{os.sep}{EXPERIMENT_ID}{os.sep}primary-model.zip'
TRAIN_WANN = False
USE_WANN = False
VISUALIZE_WANN = False
RENDER_TEST_GIFS = True
NUM_TRAIN_STEPS = 20000
############################################