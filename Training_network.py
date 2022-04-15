# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:39:40 2022

@author: hk01
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

torch.set_printoptions(linewidth=120)


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item() 

# torch.set_grad_enabled(False) # turn off the dynamic graph/gradient tracking feature in Python which uses memory
torch.set_grad_enabled(True) # turn on

print(torch.__version__)
print(torchvision.__version__)

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
    
train_set = torchvision.datasets.FashionMNIST(
    root='./data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

""" A Single batch
network =Network()
# Loading the training set from Fashion MNIST dataset

# Loading the dataset and assign it to a variable which can be inspected
train_loader = torch.utils.data.DataLoader(train_set
    ,batch_size=100
)
batch=next(iter(train_loader))
images,labels=batch # sequence/list unpacking

preds=network(images)


# Now calculate the loss
loss = F.cross_entropy(preds, labels)
loss.item()

# Now calculate the Gradients
print(network.conv1.weight.grad)
loss.backward() # calculating the gradients
network.conv1.weight.grad.shape

# Optimising the weights
optimiser = optim.Adam(network.parameters(), lr= 0.01)
loss.item()
get_num_correct(preds,labels)
optimiser.step() # updating the weights

preds=network(images)
loss=F.cross_entropy(preds, labels)
loss.item()
count_number_correct(preds,labels)
print('loss1:',loss.item())
optimiser.step() # updating the weights
preds=network(images)
loss=F.cross_entropy(preds, labels)
loss.item()
print('loss2:',loss.item())
""" 

# Running 5 epochsof all batches
network=Network()
train_loader=torch.utils.data.DataLoader(train_set, batch_size=100)
optimiser= optim.Adam(network.parameters(), lr=0.01)

for epoch in range(5):    
    total_loss=0
    total_correct=0
    for batch in train_loader: # Get batch
        images, labels = batch
    
        preds = network(images) #pass batch
        loss = F.cross_entropy(preds,labels)
    
        optimiser.zero_grad() # makes the gradient tensor zero not to add them up every time
        loss.backward() # Calcualte Gradient
        optimiser.step() # Update Weights
        
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    
    print("epoch: ",0,"Correct: ",total_correct,"loss: ",total_loss)

total_correct/len(train_set)

# plotting confusion matrices
def get_all_preds(model, loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images, labels = batch

        preds = model(images)
        all_preds = torch.cat(
            (all_preds, preds)
            ,dim=0
        )
    return all_preds

with torch.no_grad(): # this turns off the Gradient/vectore tracing option saving memory locally
    prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=10000)
    train_preds = get_all_preds(network, prediction_loader)
    
preds_correct = get_num_correct(train_preds, train_set.targets)

print('total correct:', preds_correct)
print('accuracy:', preds_correct / len(train_set))

stacked = torch.stack(
    (
        train_set.targets
        ,train_preds.argmax(dim=1)
    )
    ,dim=1
)
cmt = torch.zeros(10,10, dtype=torch.int64)
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
    


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = confusion_matrix(train_set.targets, train_preds.argmax(dim=1))

plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, train_set.classes)

    
    
    









