# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:17:12 2022

@author: hk01
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import pandas as pd
import time
import json

from itertools import product
from collections import namedtuple
from collections import OrderedDict

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
                
        # self.add_graph(self.network, images.to(getattr(run,'device', 'cpu')))        
        
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
        # results['num workers'] = 1
        for k,v in self.run_params._asdict().items(): results[k] = v
        self.run_data.append(results)   
        
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]
        
    def track_num_correct(self, preds, labels):
        self.epoch_num_correct += self._get_num_correct(preds, labels)
        
    @torch.no_grad()
    def _get_num_correct(self, preds, labels):
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
loader = DataLoader(train_set, batch_size=len(train_set))
data = next(iter(loader))
data[0].mean(), data[0].std()


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

## Now loading again and normalising the dataset
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
        ,shuffle = [True]
        ,device=['cuda','cpu']
        ,trainset = ['normal', 'not_normal'] 
    )

m = RunManager()
for run in RunBuilder.get_runs(params):
    
    # network=Network()
    # loader=DataLoader(train_set, batch_size=run.batch_size, shuffle=run.shuffle)

    device=torch.device(run.device)
    network=Network().to(device)
    loader=DataLoader(trainsets[run.trainset], batch_size=run.batch_size, shuffle=run.shuffle)
    optimiser= optim.Adam(network.parameters(), lr=run.lr)
    
    m.begin_run(run, network, loader)
    for epoch in range(5):
        m.begin_epoch()
        for batch in loader:
            
            
            # images=batch[0]
            # labels=batch[1]
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


# pd.Dataframe.from_dict(m.run_data, orient='columns').sort_values('epoch duration')
pd.DataFrame.from_dict(m.run_data).sort_values('accuracy', ascending=False)






