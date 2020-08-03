from collections import namedtuple
from extern.wann import wann_train as wt
import gym
import numpy as np

Game = namedtuple('Game', ['env_name', 'time_factor', 'actionSelect',
  'input_size', 'output_size', 'layers', 'i_act', 'h_act',
  'o_act', 'weightCap','noise_bias','output_noise','max_episode_length','in_out_labels'])

games = {}

# See reference to WANN extern wann/domain/config.py for reference config
cartpole_swingup = Game(env_name='CartPoleSwingUp_Hard',
  actionSelect='all', # all, soft, hard
  input_size=5,
  output_size=1,
  time_factor=0,
  layers=[5, 5],
  i_act=np.full(5,1),
  h_act=[1,2,3,4,5,6,7,8,9,10],
  o_act=np.full(1,1),
  weightCap=2.0,
  noise_bias=0.0,
  output_noise=[False, False, False],
  max_episode_length = 200,
  in_out_labels = ['x','x_dot','cos(theta)','sin(theta)','theta_dot',
                   'force']
)

games['swingup'] = cartpole_swingup


def balance(args):
    wt.init_games_config(games)

    # TODO: add flg here to determine if pre-training is needed
    # Train WANN feature extractor
    wt.run(args)

    # TODO: register custom wrapped env here

    # TODO: remove basic defaults once README.md is updated

    #TODO: observation wrap environment
    #TODO: register wrapped environment
    # this will use the WANN as feature abstractor from obs with final linear layer
    # to map outputs to expected input of agent
    # this will look like call(env, wann, n_out)

    # a similar stragety can also be employed for custom models that baselines uses internally

    # arguments passthrough

    # TODO: visualize and compare experiment results
    # Score results as line graph for returns over games, scores over games including mean, median, max
    # horizontal bar graph comparing scores
    # heatmap of various hyperparameter configurations in champion network(s)

    # ablation studies and dendrite graphs of variable components along with results
    # video clip play outputs
    # table of results for direct comparison

    print('agent training complete')
