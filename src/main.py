from task import cartpole, bipedal_walker
import argparse
from datetime import datetime
import sys
import multiprocessing as mp
import os
import config


# TODO: proper logging
def main():
    if not os.path.isdir(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)

    if config.TASK in ['wann-cartpolebalance-v1']:
        cartpole.balance()
    if config.TASK in ['BipedalWalker-v2']:
        bipedal_walker.walk()
    else:
        raise Exception('No implemented environment found. Please refer to list of implemented environments in README')


if __name__ == '__main__':
    main()
