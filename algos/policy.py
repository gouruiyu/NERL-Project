"""
Simple MLP policy for prey agent
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class MlpPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, n_hidden_layer, hidden_size, init, activation='relu', backprop=False):
        super(MlpPolicy, self).__init__()
        # state space dimension
        self.state_dim = state_dim
        # action space dimension
        self.action_dim = action_dim
        # number of hidden layers
        self.n_hidden_layer = n_hidden_layer
        # size of hidden layers
        self.hidden_size = hidden_size
        # activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        else:
            raise NotImplementedError
        # input layer
        self.input_layer = nn.Linear(self.state_dim, self.hidden_size)
        # hidden layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_hidden_layer)])
        # output layer
        self.output_layer = nn.Linear(self.hidden_size, self.action_dim)
        # action & reward memory
        self.action_memory = []
        self.reward_memory = []
        # whether to backpropagate or not
        self.backprop = backprop
        if not self.backprop:
            self.eval()
        
        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight)
                elif init == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: np.ndarray):
        # forward pass
        state = torch.from_numpy(state).float()
        x = self.activation(self.input_layer(state))
        for i in range(self.n_hidden_layer):
            x = self.activation(self.hidden_layers[i](x))
        x = self.output_layer(x)
        return F.softmax(x, dim=1)

    def act(self, states, deterministic=False):
        action_probs = self.forward(states)
        dist = Categorical(action_probs)
        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()
        # actions = np.argmax(self.forward(states).detach().numpy(), axis=1)
        return actions.detach().numpy()