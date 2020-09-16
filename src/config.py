import os
import logging.config
import structlog
from structlog import processors, stdlib, threadlocal, configure

VERSION_NUM = 1000

# GLOBAL CONFIGURABLE PARAMETERS
# DEFAULT CONFIGURATION
############################################
TASK = 'bipedal-walker'
EXPERIMENT_ID = f'wann-ppo-bipedalwalker-v3-{VERSION_NUM}'
SEED = 0  # high level seed for all experiments
USE_PREV_EXPERIMENT = True
TRAIN_WANN = False
USE_WANN = False
VISUALIZE_WANN = False
RENDER_TEST_GIFS = True
NUM_TRAIN_STEPS = 1000
############################################

logging.config.dictConfig(
    dict(
        version=1,
        formatters=dict(
            jsonformat={
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '%(message)s %(pathname)s %(lineno)d'
            },
        ),
        handlers=dict(
            stdout={
                'class': 'logging.StreamHandler',
                'formatter': 'jsonformat'
            },
            file={
                'class': 'logging.FileHandler',
                'filename': f'result{os.sep}{EXPERIMENT_ID}{os.sep}log{os.sep}alg-step{os.sep}alg-performance.log',
                'mode': 'w',
                'formatter': 'jsonformat',
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
        stdlib.add_logger_name,
        stdlib.add_log_level,
        stdlib.filter_by_level,
        processors.TimeStamper(fmt='iso'),
        stdlib.PositionalArgumentsFormatter(),
        processors.StackInfoRenderer(),
        stdlib.render_to_log_kwargs,
        processors.format_exc_info,
        processors.UnicodeDecoder()
    ],
    wrapper_class=stdlib.BoundLogger,
    context_class=threadlocal.wrap_dict(dict),
    logger_factory=stdlib.LoggerFactory()
)


def log():
    return structlog.getLogger('default')
