from main import main
import config


def run():
    # TODO: clean up configuration for large scale automated tests with yaml
    '''
    CARTPOLE SETUP TEST
    '''
    config.TASK = 'cartpole-balance'
    config.EXPERIMENT_ID = 'wann-cartpolebalance-v1'
    config.SEED = 0  # high level seed for all experiments
    config.USE_PREV_EXPERIMENT = False
    config.TRAIN_WANN = True
    config.USE_WANN = True
    config.VISUALIZE_WANN = False
    config.RENDER_TEST_GIFS = True
    config.NUM_TRAIN_STEPS = 30
    main()

    config.TASK = 'cartpole-balance'
    config.EXPERIMENT_ID = 'no-wann-cartpolebalance-v1'
    config.SEED = 0  # high level seed for all experiments
    config.USE_PREV_EXPERIMENT = False
    config.TRAIN_WANN = False
    config.USE_WANN = False
    config.VISUALIZE_WANN = False
    config.RENDER_TEST_GIFS = True
    config.NUM_TRAIN_STEPS = 30
    main()


    '''
    BIPEDAL WALKER SETUP TEST
    '''
    # config.TASK = 'cartpole-balance'
    # config.EXPERIMENT_ID = 'wann-cartpolebalance-v1'
    # config.SEED = 0  # high level seed for all experiments
    # config.USE_PREV_EXPERIMENT = False
    # config.TRAIN_WANN = True
    # config.USE_WANN = True
    # config.VISUALIZE_WANN = False
    # config.SHOW_TESTS = True
    # config.RENDER_TEST_GIFS = True
    # config.NUM_TRAIN_STEPS = 30
    # config.RENDER_TEST_GIFS = True
    # main()


if __name__ == '__main__':
    run()
