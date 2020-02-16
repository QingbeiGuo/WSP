#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def train_loader(path, batch_size=32, num_workers=1, pin_memory=True):
    transform_train = transforms.Compose([
        #transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])    
    
    return data.DataLoader(
        datasets.MNIST(root=path,                              
                       train=True,
                       download=False,
                       transform=transform_train),
        batch_size=batch_size,                                  
        shuffle=True,                                          
        num_workers=num_workers,                               
        pin_memory=pin_memory)

def test_loader(path, batch_size=32, num_workers=1, pin_memory=True):
    transform_test = transforms.Compose([
        #transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])    
    
    return data.DataLoader(
        datasets.MNIST(root=path,                              
                       train=False,
                       download=False,
                       transform=transform_test),
        batch_size=batch_size,                                  
        shuffle=False,                                          
        num_workers=num_workers,                                
        pin_memory=pin_memory)