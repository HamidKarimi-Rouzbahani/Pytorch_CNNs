# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:16:12 2022

@author: hk01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict

import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple


torch.set_printoptions(linewidth=150)


# # Now let's make the network which we built in this course using sequential technique
# torch.manual_seed(50)
# network1 = nn.Sequential(
#       nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
#     , nn.ReLU()
#     , nn.MaxPool2d(kernel_size=2, stride=2)
#     , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
#     , nn.ReLU()
#     , nn.MaxPool2d(kernel_size=2, stride=2)
#     , nn.Flatten(start_dim=1)  
#     , nn.Linear(in_features=12*4*4, out_features=120)
#     , nn.ReLU()
#     , nn.Linear(in_features=120, out_features=60)
#     , nn.ReLU()
#     , nn.Linear(in_features=60, out_features=10)
# )

# # Now adding batch-norm function which normalises the data 
# torch.manual_seed(50)
# network2 = nn.Sequential(
#       nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
#     , nn.ReLU()
#     , nn.MaxPool2d(kernel_size=2, stride=2)
#     , nn.BatchNorm2d(6)
#     , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
#     , nn.ReLU()
#     , nn.MaxPool2d(kernel_size=2, stride=2)
#     , nn.Flatten(start_dim=1)  
#     , nn.Linear(in_features=12*4*4, out_features=120)
#     , nn.ReLU()
#     , nn.BatchNorm1d(120)
#     , nn.Linear(in_features=120, out_features=60)
#     , nn.ReLU()
#     , nn.Linear(in_features=60, out_features=10)
# )


## Now preparing the runner as before


class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_num_correct = 0
        self.epoch_start_time = None
    
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
    
        self.network = None
        self.loader = None
                
    def begin_run(self, run, network, loader):   
        self.run_start_time = time.time()    
        self.run_params = run
        self.run_count += 1
        
        self.network = network
        
        self.loader = loader
        
        
        images, labels = next(iter(self.loader))
        # grid = torchvision.utils.make_grid(images)  
                
    def end_run(self):
        self.epoch_count = 0        
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_num_correct = 0
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_num_correct / len(self.loader.dataset)
        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results['loss'] = loss
        results["accuracy"] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)   
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self.get_num_correct(preds, labels)
        
    @torch.no_grad()
    def get_num_correct(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()   

    def save(self, fileName):
    
        pd.DataFrame.from_dict(
            self.run_data, orient='columns'
        ).to_csv(f'{fileName}.csv')
    
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)    
    
    
class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs
    

class NetworkFactory():
    @staticmethod
    def get_network(name):
        if name == 'normal':
            torch.manual_seed(50)
            return nn.Sequential(
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
        elif name == 'batch_norm':
             torch.manual_seed(50)
             return nn.Sequential(
                  nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
                , nn.ReLU()
                , nn.MaxPool2d(kernel_size=2, stride=2)
                , nn.BatchNorm2d(6)
                , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
                , nn.ReLU()
                , nn.MaxPool2d(kernel_size=2, stride=2)
                , nn.Flatten(start_dim=1)  
                , nn.Linear(in_features=12*4*4, out_features=120)
                , nn.ReLU()
                , nn.BatchNorm1d(120)
                , nn.Linear(in_features=120, out_features=60)
                , nn.ReLU()
                , nn.Linear(in_features=60, out_features=10)
            ) 
        elif name == 'no_max_pool':
            torch.manual_seed(50)
            return nn.Sequential(
                    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
                , nn.ReLU()
                # , nn.MaxPool2d(kernel_size=2, stride=2)
                , nn.BatchNorm2d(6)
                , nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
                , nn.ReLU()
                # , nn.MaxPool2d(kernel_size=2, stride=2)
                , nn.Flatten(start_dim=1)  
                , nn.Linear(in_features=12*20*20, out_features=120)
                , nn.ReLU()
                , nn.BatchNorm1d(120)
                , nn.Linear(in_features=120, out_features=60)
                , nn.ReLU()
                , nn.Linear(in_features=60, out_features=10)
            )
        else:
            return None
        
    
train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)
        
# normalising dataset

# easy way: when you can fit the whole datset into memory
# loader = DataLoader(train_set, batch_size=len(train_set))
# data = next(iter(loader))
# data[0].mean(), data[0].std()

# hard way: when the datset is big
loader = DataLoader(train_set, batch_size=1000)
num_of_pixels = len(train_set) * 28 * 28
total_sum = 0
for batch in loader: total_sum += batch[0].sum()
mean = total_sum / num_of_pixels

sum_of_squared_error = 0
for batch in loader: 
    sum_of_squared_error += ((batch[0] - mean).pow(2)).sum()
std = torch.sqrt(sum_of_squared_error / num_of_pixels)

# Now loading again and normalising the dataset
train_set_normal = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
          transforms.ToTensor()
        , transforms.Normalize(mean, std)
    ])
)

trainsets = {
    'not_normal': train_set
    ,'normal': train_set_normal
}


params = OrderedDict(
        lr = [.01]
        ,batch_size = [1000]
        ,device=['cuda']
        ,trainset = ['normal', 'not_normal']
        , network = ['normal', 'batch_norm', 'no_max_pool']
    )

m = RunManager()
for run in RunBuilder.get_runs(params):
    
    device=torch.device(run.device)
    network = NetworkFactory.get_network(run.network).to(device) # Factory call
    loader=DataLoader(trainsets[run.trainset], batch_size=run.batch_size)
    optimiser= optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:
            
            images=batch[0].to(device)
            labels=batch[1].to(device)
            preds=network(images) #pass batch
            loss = F.cross_entropy(preds,labels)        
            optimiser.zero_grad() # makes the gradient tensor zero not to add them up every time
            loss.backward() # Calcualte Gradient
            optimiser.step() # Update Weights
        
            m.track_loss(loss,batch)
            m.track_num_correct(preds, labels)
        
        m.end_epoch()
    m.end_run()
m.save('results')


pd.DataFrame.from_dict(m.run_data).sort_values('accuracy', ascending=False)







