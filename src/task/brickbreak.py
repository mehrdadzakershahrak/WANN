import gym
from baselines import run as B


def brickbreak_ram(args):
    # TODO: register custom wrapped env here

    # TODO: remove basic defaults once README.md is updated

    #TODO: observation wrap environment
    #TODO: register wrapped environment
    # this will use the WANN as feature abstractor from obs with final linear layer
    # to map outputs to expected input of agent
    # this will look like call(env, wann, n_out)

    # a similar stragety can also be employed for custom models that baselines uses internally

    # TODO: visualize and compare experiment results
    # Score results as line graph for returns over games, scores over games including mean, median, max
    # horizontal bar graph comparing scores
    # heatmap of various hyperparameter configurations in champion network(s)

    # ablation studies and dendrite graphs of variable components along with results
    # video clip play outputs
    # table of results for direct comparison

    print('agent training complete')