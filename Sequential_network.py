# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:38:09 2022

@author: hk01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import math

from collections import OrderedDict

import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple


torch.set_printoptions(linewidth=150)

train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

image, label = train_set[0]

in_features = image.numel()
out_features = math.floor(in_features / 2)
out_classes = len(train_set.classes)



# First way of making sequential networks
torch.manual_seed(50)
network1 = nn.Sequential(
    nn.Flatten(start_dim=1)
    ,nn.Linear(in_features, out_features)
    ,nn.Linear(out_features, out_classes)
)

# Second way of making sequential networks
torch.manual_seed(50)
layers = OrderedDict([
    ('flat', nn.Flatten(start_dim=1))
   ,('hidden', nn.Linear(in_features, out_features))
   ,('output', nn.Linear(out_features, out_classes))
])

network2 = nn.Sequential(layers)

# Third way of making sequential networks

torch.manual_seed(50)
network3 = nn.Sequential()
network3.add_module('flat', nn.Flatten(start_dim=1))
network3.add_module('hidden', nn.Linear(in_features, out_features))
network3.add_module('output', nn.Linear(out_features, out_classes))



# Now let's make the network which we built in this course using sequential technique
torch.manual_seed(50)
network1 = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
    , nn.ReLU()
    , nn.MaxPool2d(kernel_size=2, stride=2)
    , nn.Flatten(start_dim=1)  
    , nn.Linear(in_features=12*4*4, out_features=120)
    , nn.ReLU()
    , nn.Linear(in_features=120, out_features=60)
    , nn.ReLU()
    , nn.Linear(in_features=60, out_features=10)
)
