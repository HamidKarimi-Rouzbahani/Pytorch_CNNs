# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 17:22:25 2022

@author: hk01
"""
import torch
import numpy as np
torch. cuda. is_available()

data=np.array([1,2,3])

t1=torch.Tensor(data) # constructor: separate memory from numpy

t2=torch.tensor(data) # factory function: separate memory from numpy

t3=torch.as_tensor(data) # factory function: any array_like data structure; same memory with numpy array so if you change the data after this line, t3 also changes

t4=torch.from_numpy(data) # factory function: only from numpy arrays; same memory with numpy array so if you change the data after this line, t4 also changes

print(t1)
print(t2)
print(t3)
print(t4)

print(t1.dtype)
print(t2.dtype)
print(t3.dtype)
print(t4.dtype)

torch.get_default_dtype()

t_temp=torch.tensor(np.array([1,2,3]), dtype=torch.float64)
print(t_temp.dtype)

# reshaping

t= torch.tensor([
                [1,1,1,1],
                [2,2,2,2],
                [3,3,3,3]
                ], dtype=torch.float32)

print(t.size())

print(len(t.shape))

print(torch.tensor(t.shape).prod())

print(t.numel())

print(t.reshape(6,2))

print(t.reshape(1,12).squeeze().shape)

print(t.reshape(1,12).unsqueeze(dim=0).shape)

def flatten (t):
    t=t.reshape(1,-1)
    t=t.squeeze()
    return t

print (t)
t_flattened=flatten(t)
print(t_flattened)

# Concatenating

t1=torch.tensor([
    [1,2],
    [3,4]
    ])

t2=torch.tensor([
    [3,4],
    [5,6]
                ])

t_cat=torch.cat((t1,t2), dim=1)

print(t_cat)
print(t_cat.shape)












