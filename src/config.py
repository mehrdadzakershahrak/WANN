import os

# GLOBAL CONFIGURABLE PARAMETERS
# TODO: make GLOBAL CONFIGS YML driven
############################################
TASK = os.getenv('RUN_TASK', 'cartpole-balance')
SEED = os.getenv('RUN_SEED', 0)  # high level seed for all experiments
USE_PREV_EXPERIMENT = False
TRAIN_WANN = True
USE_WANN = True
VISUALIZE_WANN = True
SHOW_TESTS = True
RENDER_TEST_GIFS = True
############################################
