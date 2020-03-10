import torch
from torch import nn
import numpy as np
from torch.distributions import Categorical

def network_factory(in_size, num_actions, env):
    network = nn.Sequential(nn.Linear(in_size, 32), nn.ReLU(), nn.Linear(32, num_actions), nn.Softmax(dim=-1))
    return network


class PolicyNetwork(nn.Module):
    def __init__(self, network):
        super(PolicyNetwork, self).__init__()
        self.network = network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action_probs = self.network(state)
        return Categorical(action_probs)

    def get_action(self, state):
        category = self.forward(state)
        return category.sample().item()


class ValueNetwork(nn.Module):
    def __init__(self, in_size):
        super(ValueNetwork, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network_v = nn.Sequential(nn.Linear(in_size, 32, bias=True), nn.ReLU(), nn.Linear(32, 1, bias=True))

    def forward(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        return self.network_v(state)

    def get_value(self, state):
        return self.forward(state).item()
