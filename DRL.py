import torch
import math
import random

class DRL(torch.nn.Module):
    """
    DRL (Deep Reinforcement Learning) class extends PyTorch's Module class and implements a simple
    feed-forward neural network for reinforcement learning.
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        """
        Initialize DRL object with network parameters and training parameters.

        input_size: Number of input features.
        hidden_size: Number of neurons in hidden layers.
        output_size: Number of output neurons.
        learning_rate: Learning rate for the optimizer.
        """
        super(DRL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Define a simple feed-forward network with two hidden layers and ReLU activation
        self.mapping = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_size, hidden_size), 
            torch.nn.ReLU(), 
            torch.nn.Linear(hidden_size, output_size)
        )

        self.apply(self.__class__.weights_init)  # Initialize network weights
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)  # Adam optimizer
        self.criterion = torch.nn.MSELoss()  # Mean Squared Error loss for regression problem

        # Parameters for epsilon-greedy exploration
        self.steps = 0
        self.buffer = []
        self.epsi_low = 0.05
        self.epsi_high = 0.9
        self.gamma = 0.8
        self.decay = 200
        self.capacity = 10000
        self.batch_size = 64

    def weights_init(m):
        """
        Initialize network weights. 

        m: Network module.
        """
        if m.__class__.__name__.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)  # Initialize weights with normal distribution
            torch.nn.init.constant_(m.bias.data, val=0.0)  # Initialize bias with constant

    def explore(self, state):
        """
        Implement epsilon-greedy exploration strategy. 

        state: Current state of the agent.
        """
        self.steps += 1
        epsilon = self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay))
        if random.random() < epsilon:
            action = random.randrange(self.output_size)  # Random action (exploration)
        else:
            state = torch.tensor(state, dtype=torch.float).view(1,-1)
            action = torch.argmax(self.mapping(state)).item()  # Best action (exploitation)
        return action

    def remember(self, *transition):
        """
        Store transition in memory buffer for experience replay.

        transition: Tuple (state, action, reward, new_state).
        """
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)  # Remove oldest transition if buffer is full
        self.buffer.append(transition)  # Add new transition to buffer

        def rethink(self):
        """
        Implement experience replay and network update.
        """
        # Only update the network if there are enough transitions in the buffer
        if len(self.buffer) < self.batch_size:
            return

        # Randomly sample a batch of transitions from the buffer
        transitions = random.sample(self.buffer, self.batch_size)
        state_old, action_now, reward_now, state_new = zip(*transitions)

        # Convert the transitions to PyTorch tensors for computation
        state_old = torch.tensor(state_old, dtype=torch.float)
        action_now = torch.tensor(action_now, dtype=torch.long).view(self.batch_size, -1)
        reward_now = torch.tensor(reward_now, dtype=torch.float).view(self.batch_size, -1)
        state_new = torch.tensor(state_new, dtype=torch.float)

        # Compute the target Q-values using the reward and the maximum Q-value of the new state
        y_true = reward_now + self.gamma * torch.max(self.mapping(state_new).detach(), dim=1)[0].view(self.batch_size, -1)

        # Compute the predicted Q-values from the old state and the executed action
        y_pred = self.mapping(state_old).gather(1, action_now)

        # Compute the loss between the target and predicted Q-values
        loss = self.criterion(y_pred, y_true)

        # Backpropagate the loss through the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
