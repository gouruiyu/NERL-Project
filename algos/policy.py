"""
Simple MLP policy for prey agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class RandomBaselinePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def act(self, obs):
        return np.random.randint(0, 5)


class MlpPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_layer, hidden_size, init, activation='relu', backprop=False):
        super(MlpPolicy, self).__init__()
        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'tanh':
            activation = nn.Tanh
        elif activation == 'sigmoid':
            activation = nn.Sigmoid
        else:
            raise NotImplementedError
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_size))
        self.layers.append(activation())
        for _ in range(n_hidden_layer):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_size, action_dim))
        
        # Xavier initialization
        for m in self.layers:
            if isinstance(m, nn.Linear):
                if init == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                elif init == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                elif init == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            
        # freeze weights
        if not backprop:
            for param in self.parameters():
                param.requires_grad = False
                
    def add_noise(self, std=0.1):
        for param in self.parameters():
            param.data += torch.randn(param.data.size()) * std
            
    def forward(self, x: np.ndarray):
        x = torch.from_numpy(x).float()
        # forward pass
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=-1)

    def act(self, states, deterministic=True):
        action_probs = self.forward(states)
        if deterministic:
            actions = torch.argmax(action_probs) # FIXME: does not work for batch, refactor
        else:
            dist = Categorical(action_probs)
            actions = dist.sample()
        # actions = np.argmax(self.forward(states).detach().numpy(), axis=1)
        return actions.detach().numpy()
    
    def save(self, path):
        torch.save(self.state_dict(), path)