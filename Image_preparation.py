# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:06:04 2022

@author: hk01
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)


# Loading the training set from Fashion MNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

# Loading the dataset and assign it to a variable which can be inspected
train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=10
)

len(train_set) # check the size of the dataset

train_set.train_labels # check labels

train_set.train_labels.bincount() # check balance of the dataset


# access one image
sample=next(iter(train_set))
image,label=sample # sequence/list unpacking
plt.imshow(image.squeeze(), cmap='gray')
print('label:', label)

# access a batch of images
batch=next(iter(train_loader))
images,labels=batch # sequence/list unpacking
grid = torchvision.utils.make_grid(images, nrow=10)

plt.figure(figsize=(15,15))
plt.imshow(np.transpose(grid, (1,2,0)))

print('labels:', labels)