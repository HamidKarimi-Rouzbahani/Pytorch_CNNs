# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:28:32 2022

@author: hk01
"""
import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) #in_channels=1 refers to the number of colour channels    
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # implement the forward pass
        
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t=F.relu(t)
        t= F.max_pool2d(t,kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        t = self.conv2(t)
        t=F.relu(t)
        t= F.max_pool2d(t,kernel_size=2, stride=2)


        # (4) hidden linear layer
        t= t.reshape(-1,12*4*4)
        t=self.fc1(t)
        t=F.relu(t)
        
        # (5) hidden linear layer
        t=self.fc2(t)
        t=F.relu(t)
        
        # (6) hidden linear layer
        t=self.out(t)
        # t=F.softmax(t,dim=1)  # we needed to use softmax if we were not going to use inter-class entropy for training which applied sfotmax automatically
                        
        return t
    
network =Network()

network.fc2

network.conv2.weight

network.conv2.weight[0].shape

network.fc1.weight[0].shape

# checking all parameters: 1
for param in network.parameters():
    print(param.shape)
    
#checking all parameters: 2
for name, param in network.named_parameters():
    print(name, '\t\t', param.shape)    
    
# generating some example linear layers
fc = nn.Linear(in_features=4, out_features=3)    
t=torch.tensor([1,2,3,4], dtype=torch.float32)
output = fc(t)