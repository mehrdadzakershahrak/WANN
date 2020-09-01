import os

VERSION_NUM = 4

# GLOBAL CONFIGURABLE PARAMETERS
# DEFAULT CONFIGURATION
############################################
TASK = 'bipedal-walker'
EXPERIMENT_ID = f'wann-bipedalwalker-v3-{VERSION_NUM}'
PREV_EXPERIMENT_PATH = f'result{os.sep}artifact{os.sep}{EXPERIMENT_ID}{os.sep}primary-model.zip'
SEED = 0  # high level seed for all experiments
USE_PREV_EXPERIMENT = False
START_FROM_LAST_RUN = False
TRAIN_WANN = True
USE_WANN = True
VISUALIZE_WANN = False
RENDER_TEST_GIFS = True
NUM_TRAIN_STEPS = 1000
############################################


# config.TASK = 'bipedal-walker'
# config.EXPERIMENT_ID = f'wann-bipedalwalker-v3-{VERSION_NUM}'
# config.SEED = 0  # high level seed for all experiments
# config.TRAIN_WANN = False
# config.USE_WANN = True
# config.USE_PREV_EXPERIMENT = True
# config.PREV_EXPERIMENT_PATH = f'result{os.sep}artifact{os.sep}wann-bipedalwalker-v3-1{os.sep}primary-model.zip'
# config.RENDER_TEST_GIFS = True
# config.NUM_TRAIN_STEPS = 1000
# main()
#
# config.TASK = 'bipedal-walker'
# config.EXPERIMENT_ID = f'no-wann-bipedalwalker-v3-{VERSION_NUM}'
# config.SEED = 0  # high level seed for all experiments
# config.TRAIN_WANN = False
# config.USE_WANN = False
# config.RENDER_TEST_GIFS = True
# config.NUM_TRAIN_STEPS = 1000
# main()
