import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, h1_units=64, h2_units=64):
        """Initialize a NN model.

        Parameters:
            - state_size (int): no. of state dimensions (defines input layer)
            - acion_size (int): no. of action dimensions (defines output layer)
            - seed (int):       pytorch random number generator seed
            - h1_units (int):   neurons in hidden layer 1
            - h2_units (int):   neurons in hidden layer 2
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, h1_units)
        self.fc2 = nn.Linear(h1_units, h2_units)
        self.fc3 = nn.Linear(h2_units, action_size)


    def forward(self, x):
        """
        Maps state input x to action values
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
