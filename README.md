# Weight Agostic Neural Network

## Overview
This project comprises implementations of two classes, DRL (Deep Reinforcement Learning) and WAN (Weight Agnostic Neural Network), along with three main functions: drl, wan, and tpj.

DRL Class
DRL is a class for Deep Reinforcement Learning. It uses PyTorch to create a neural network and includes various methods for exploration, remembering state transitions, and updating the network weights based on the transitions in the memory buffer.

WAN Class
WAN stands for Weight Agnostic Neural Network. It is initialized with a shared weight value. The architecture of the network is defined in the constructor, with a certain number of hidden nodes, input nodes, and output nodes. It also includes a weight vector, initially set to zero, and a bias. The class contains methods for setting the weights, tuning the weights, applying various activation functions, and calculating the output for a given input.

## Main Functions
drl Function
The drl function uses a DRL object to simulate an environment (a 'CartPole' environment in this case) over 1000 epochs.

wan Function
The wan function simulates the 'CartPoleSwingUpEnv' environment using a WAN object with a shared weight value of -1.5. It iterates over 20 epochs, and after the 10th epoch, it adjusts the weights in the WAN.

tpj Function
The tpj function is currently a placeholder. The comments suggest that this function is intended to implement some kind of evolutionary algorithm for neural networks, including steps for initializing a population, evaluating and ranking networks, and creating new networks from the best performing ones. This function is yet to be implemented.

## Usage
Please note that the script won't run as it is due to a number of issues, including missing imports (e.g., CartPoleEnv, CartPoleSwingUpEnv) and some undefined functions/classes (e.g., DRL). Also, the tpj function is currently empty and would need to be implemented.

## Contributions
Contributions are welcome! Feel free to add improvements, bugfixes, or new features via pull requests.