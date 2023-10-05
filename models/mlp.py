import torch
import torch.nn as nn
import torch.nn.functional as F
"""
Initialize MLP. 
Args.
@param input_size - The size of the input layer.
@param output_size - The size of the output layer.
@param hidden_sizes - The size of the hidden layers.
@param layers - The number of layers. Default is 1.
@param dropout - The dropout value. Default is 0. ( 0 by default )
@param normal - Whether to use batch Normalization (True by default)
"""
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes = [], dropout = 0, normal = True):
        self.layers = nn.ModuleList
        for i,hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(nn.Linear(input_size,hidden_size))
            else:
                self.layers.apply(nn.Linear[hidden_sizes[i-1],hidden_size])
            self.layers.append(nn.ReLU())
            if normal:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            if dropout:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_sizes[-1],output_size))
    
    def forward(self, x):
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        return x
