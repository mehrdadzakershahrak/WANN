import os
import logging.config
import structlog
from structlog import processors, stdlib, threadlocal, configure

VERSION_NUM = 1
EXPERIMENT_ID = f'wann-bipedal-use-current-{VERSION_NUM}'  # TODO: DRY ME UP
USE_WANN = True  # TODO: DRY ME UP

run_config = dict(
    TASK='bipedal-walker',
    EXPERIMENT_ID=EXPERIMENT_ID,
    SEED=0,  # high level seed for all experiments
    USE_PREV_EXPERIMENT=False,
    PREV_EXPERIMENT_PATH='prev-run',
    TRAIN_WANN=True,
    USE_WANN=USE_WANN,
    VISUALIZE_WANN=False,
    RENDER_TEST_GIFS=False,
    NUM_EPOCHS=500,
    DESCRIPTION='''
    This experiment implements WANN with the SAC critic sampled from the replay buffer
    
    bipedal new config test
    '''
)

performance_log_path = f'result{os.sep}{run_config["EXPERIMENT_ID"]}{os.sep}log{os.sep}alg-step{os.sep}'
if not os.path.isdir(performance_log_path):
    os.makedirs(performance_log_path)

logging.config.dictConfig(
    dict(
        version=1,
        handlers=dict(
            file={
                'class': 'logging.FileHandler',
                'filename': performance_log_path + 'alg-performance.log',
                'mode': 'w',
                'formatter': 'jsonformat',
            },
            stdout={
                'class': 'logging.StreamHandler',
                'formatter': 'jsonformat'
            }
        ),
        formatters=dict(
            jsonformat={
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(message)s'
            },
        ),
        loggers={
            '': {
                'handlers': ['stdout', 'file'],
                'level': logging.INFO
            }
        },
        disable_existing_loggers=True
    )
)

configure(
    processors=[
        processors.TimeStamper(fmt='iso'),
        processors.format_exc_info,
        processors.StackInfoRenderer(),
        stdlib.filter_by_level,
        processors.UnicodeDecoder(),
        stdlib.render_to_log_kwargs
    ],
    context_class=threadlocal.wrap_dict(dict),
    logger_factory=stdlib.LoggerFactory(),
    wrapper_class=stdlib.BoundLogger
)


def log():
    return structlog.getLogger('default')
