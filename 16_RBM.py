import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

def convert(data):
    new_data = []
    for i in range(1, 944):
        id_movies = data[:,1][data[:,0] == i]
        id_ratings = data[:,2][data[:,0] == i]
        ratings = np.zeros(1682)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings)) 
    return new_data
    
W = torch.randn(200, 1682)
a = torch.randn(1, 200)
b = torch.randn(1, 1682)


def hidden(x):
    wx = torch.mm(x,W.t())
    activation = wx + a.expand_as(wx)
    ph = torch.sigmoid(activation)
    return ph, torch.bernoulli(ph)
     
def visible(y):
    wy = torch.mm(y,W)
    activation = wy + b.expand_as(wy)
    pv = torch.sigmoid(activation)
    return pv, torch.bernoulli(pv)
        
def train(v0, vk, ph0, phk):
    global W,a,b
    W += (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
    b += torch.sum((v0 - vk), 0)
    a += torch.sum((ph0 - phk), 0)

training_set = pd.read_csv('./u1.base', delimiter = '\t')
test_set = pd.read_csv('./u1.test', delimiter = '\t')

training_set = np.array(training_set, dtype = 'int')
test_set = np.array(test_set, dtype = 'int') 

print (max(max(training_set[:,0]), max(test_set[:,0])))
print (max(max(training_set[:,1]), max(test_set[:,1])))

training_set = convert(training_set)
test_set = convert(test_set)

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

for epoch in range(1, 11):
    train_loss = 0
    s = 0.
    for i in range(0, 943, 100):
        vk = training_set[i:i+100]
        v0 = training_set[i:i+100]
        ph0,_ = hidden(v0)
        for k in range(10):
            _,hk = hidden(vk)
            _,vk = visible(hk)
            vk[v0<0] = v0[v0<0]
        phk,_ = hidden(vk)
        train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
test_loss = 0
s = 0.
for i in range(943):
    v = training_set[i:i+1]
    vt = test_set[i:i+1]
    if len(vt[vt>=0]) > 0:
        _,h = hidden(v)
        _,v = visible(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))



