import os

VERSION_NUM = 1000

# GLOBAL CONFIGURABLE PARAMETERS
# DEFAULT CONFIGURATION
############################################
TASK = 'bipedal-walker'
EXPERIMENT_ID = f'wann-ppo-bipedalwalker-v3-{VERSION_NUM}'
PREV_EXPERIMENT_PATH = f'result{os.sep}artifact{os.sep}{EXPERIMENT_ID}{os.sep}primary-model.zip'
SEED = 0  # high level seed for all experiments
USE_PREV_EXPERIMENT = False
START_FROM_LAST_RUN = False
TRAIN_WANN = False
USE_WANN = False
VISUALIZE_WANN = False
RENDER_TEST_GIFS = True
NUM_TRAIN_STEPS = 20000
############################################