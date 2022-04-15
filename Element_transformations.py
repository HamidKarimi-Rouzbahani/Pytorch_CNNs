# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 11:48:10 2022

@author: hk01
"""

import torch
import numpy as np

t1 =torch.tensor(np.ones((4,4)),dtype=torch.int32)
t2 =2*torch.tensor(np.ones((4,4)),dtype=torch.int32)
t3 =3*torch.tensor(np.ones((4,4)),dtype=torch.int32)
print (t2)

t = torch.stack((t1,t2,t3))
t.shape

print(t)
t=t.reshape(3,1,4,4) # image, colour channel, x and y dimensions
print(t)

print(t[2][0]) # inspect some elements of the tensor


# print(t.reshape(-1)) # flattens the whole tensor

print(t.flatten(start_dim = 1)) # from which dimension start flattening (here 1 means start from colour channel)
# or we can use reshape function
# print(t.reshape(3,1,t.size(2)*t.size(3)).squeeze())



# Element-wise operations

t1=torch.tensor([[1,1],
                 [1,1],
                 ])
t2=torch.tensor([1,3])

print(t1+t2)

# using numpy broadcasting to make the two tensors equal in size
print(np.broadcast_to(t2.numpy(),t1.shape))
print (t1+t2)
print(t2.add(t1))

# now check what happens with lists
t3=[[1,1],
    [1,1]]

t4=[[2,3]]

print(t3+t4)


# now doing some comparison
print(t1.gt(2))

print(t[0][0].squeeze() <= t1)


# Reduction operations

t=torch.tensor([[0,1,0],
                [2,0,5],
                [0,4,0]],dtype=torch.float32)

t.sum().numel() <= t.numel()

t.prod()
t.std()
t.sum(dim=1)
print(t.max(dim=0))
print(t.argmax(dim=0))

print(t.mean(dim=0).tolist()) # turn output values into a list

print(t.mean(dim=0).numpy()) # turn output values into numpy array




t = torch.tensor([
    [1,0,0,2],
    [0,3,3,0],
    [4,0,0,5]
], dtype=torch.float32)

t.argmax(dim=0)









