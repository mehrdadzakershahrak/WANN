from main import main
import config

VERSION_NUM = 1


def run():
    # TODO: clean up configuration for large scale automated tests with yaml
    '''
    CARTPOLE SETUP TEST
    '''
    # config.TASK = f'cartpole-balance'
    # config.EXPERIMENT_ID = f'wann-cartpolebalance-v1-{VERSION_NUM}'
    # config.SEED = 0  # high level seed for all experiments
    # config.USE_PREV_EXPERIMENT = False
    # config.TRAIN_WANN = True
    # config.USE_WANN = True
    # config.VISUALIZE_WANN = False
    # config.RENDER_TEST_GIFS = True
    # config.NUM_TRAIN_STEPS = 30
    # main()
    #
    # config.TASK = f'cartpole-balance'
    # config.EXPERIMENT_ID = f'no-wann-cartpolebalance-v1-{VERSION_NUM}'
    # config.SEED = 0  # high level seed for all experiments
    # config.TRAIN_WANN = False
    # config.USE_WANN = False
    # config.VISUALIZE_WANN = False
    # config.RENDER_TEST_GIFS = True
    # config.NUM_TRAIN_STEPS = 30
    # main()

    '''
    BIPEDAL WALKER SETUP TEST
    '''
    config.TASK = 'bipedal-walker'
    config.EXPERIMENT_ID = f'wann-bipedalwalker-v3-{VERSION_NUM}'
    config.SEED = 0  # high level seed for all experiments
    config.TRAIN_WANN = True
    config.USE_WANN = True
    config.RENDER_TEST_GIFS = True
    config.NUM_TRAIN_STEPS = 1000
    main()

    config.TASK = 'bipedal-walker'
    config.EXPERIMENT_ID = f'no-wann-bipedalwalker-v3-{VERSION_NUM}'
    config.SEED = 0  # high level seed for all experiments
    config.TRAIN_WANN = False
    config.USE_WANN = False
    config.RENDER_TEST_GIFS = True
    config.NUM_TRAIN_STEPS = 1000
    main()


if __name__ == '__main__':
    run()
