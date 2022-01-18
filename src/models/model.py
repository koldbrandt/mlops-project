import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels=8, kernel_size = 3)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels=16 ,kernel_size = 5)
        self.fc1 = nn.Linear(16*4*4,128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.output = nn.Sequential(nn.Linear(64, 10), 
                                   nn.LogSoftmax(dim=1))
        #self.output = nn.Linear(64, 10)
        
    def forward(self, x):
        # first conv
        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            print("{}, {}, {}".format(x.shape[1], x.shape[2],x.shape[3]))
            raise ValueError('Expected each sample to have shape x,1,28,28')


        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2,stride=2)
        # second conv
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,kernel_size=2,stride=2)
        
        # Hidden layers
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer with softmax activation
        
        x = self.output(x)        
        return x

