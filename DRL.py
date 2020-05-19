import torch
import math
import random
class DRL(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(DRL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.mapping = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, output_size))
        self.apply(self.__class__.weights_init)  #TPJ
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.steps = 0
        self.buffer = []
        self.epsi_low = 0.05
        self.epsi_high = 0.9
        self.gamma = 0.8
        self.decay = 200
        self.capacity = 10000
        self.batch_size = 64

    def weights_init(m):
        if m.__class__.__name__.find('Linear') != -1:
            torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
            torch.nn.init.constant_(m.bias.data, val=0.0)
        
    def explore(self, state):
        self.steps += 1
        if random.random() < self.epsi_low + (self.epsi_high-self.epsi_low) * (math.exp(-1.0 * self.steps/self.decay)):
            action = random.randrange(self.output_size)
        else:
            state = torch.tensor(state, dtype=torch.float).view(1,-1)
            action = torch.argmax(self.mapping(state)).item()
        return action

    def remember(self, *transition):
        if len( self.buffer)==self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
        
    def rethink(self):
        if len(self.buffer) >= self.batch_size:
            state_old, action_now, reward_now, state_new = zip(*random.sample(self.buffer, self.batch_size))
            state_old = torch.tensor(state_old, dtype=torch.float)
            action_now = torch.tensor(action_now, dtype=torch.long).view(self.batch_size, -1)
            reward_now = torch.tensor(reward_now, dtype=torch.float).view(self.batch_size, -1)
            state_new = torch.tensor(state_new, dtype=torch.float)
            y_true = reward_now + self.gamma * torch.max( self.mapping(state_new).detach(), dim=1)[0].view(self.batch_size, -1)
            y_pred = self.mapping.forward(state_old).gather(1, action_now)
            loss = self.criterion(y_pred, y_true)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
