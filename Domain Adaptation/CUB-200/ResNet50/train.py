#!/usr/bin/python
# -*- coding: UTF-8 -*-

#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import torch
from torch.autograd import Variable
from torchvision import models
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import os
import math
from sklearn.cluster import KMeans  

from models.resnet.resnet import resnet50
import dataset.dataset_cub200
#import kmeans

##############################################################################################################
class GroupingFineTuner_ResNet:
	def __init__(self, train_path, test_path, model):
	    self.args = args
	    self.learningrate = self.args.learning_rate
	    self.learning_rate_decay = self.args.learning_rate_decay
	    self.momentum = self.args.momentum
	    self.weight_decay = self.args.weight_decay
	    self.train_path = self.args.train_path
	    self.test_path = self.args.test_path

	    self.train_data_loader = dataset.dataset_cub200.train_loader(self.train_path)
	    self.test_data_loader = dataset.dataset_cub200.test_loader(self.test_path)

	    self.model = model
	    self.criterion = torch.nn.CrossEntropyLoss()

	    self.accuracys1 = []
	    self.accuracys5 = []

	    self.criterion.cuda()
	    self.model.cuda()

	    for param in self.model.parameters():
	        param.requires_grad = True

	    self.model.train()

##############################################################################################################
	def train(self, epoches = -1, batches = -1):
		epoch_i = -1
		if os.path.isfile("epoch_i"):
		    epoch_i = torch.load("epoch_i")
		    print("epoch_i resume:", epoch_i)

		    self.model = torch.load("model_training_" + str(epoch_i))
		    print("model_training resume:", self.model)

		    self.accuracys1 = torch.load("accuracys1_trainning")
		    self.accuracys5 = torch.load("accuracys5_trainning")
		    print("accuracys1_trainning resume:", self.accuracys1)
		    print("accuracys5_trainning resume:", self.accuracys5)

		    self.test(0)
		else:
		    self.test()

		accuracy = 0
		for i in list(range(epoches)):
		    print("Epoch: ", i)
		    self.adjust_learning_rate(i)

		    if i <= epoch_i:
		        continue

		    optimizer = optim.SGD(self.model.parameters(), lr=self.learningrate, momentum=self.momentum, weight_decay=self.weight_decay)
		    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

		    self.train_epoch(i, batches, optimizer, scheduler)
		    cor1, cor5 = self.test()

		    #save the best model
		    if cor1 > accuracy:
		        torch.save(self.model, "model_training_m")
		        accuracy = cor1

		    torch.save(i, "epoch_i")
		    torch.save(self.model, "model_training_" + str(i))
		    torch.save(self.accuracys1, "accuracys1_trainning")
		    torch.save(self.accuracys5, "accuracys5_trainning")

	def train_epoch(self, epoch, batches, optimizer = None, scheduler = None):
		for step, (batch, label) in enumerate(self.train_data_loader):
		    if (step == batches):
		        break
		    self.train_batch(epoch, step, batch, label, optimizer, scheduler)

	def train_batch(self, epoch, step, batch, label, optimizer = None, scheduler = None):
		### Compute output
		batch,label = Variable(batch.cuda()),Variable(label.cuda())                   #Tensor->Variable
		output = self.model(batch)
		loss = self.criterion(output, label)

		if step % self.args.print_freq == 0:
		    print("Epoch-step: ", epoch, "-", step, ":", loss.data.cpu().numpy())

		### Compute gradient and do SGD step
		self.model.zero_grad()
		loss.backward()
		optimizer.step()                                                              #update parameters

	def test(self, flag = -1):
		self.model.eval()

		#correct = 0
		correct1 = 0
		correct5 = 0
		total = 0

		print("Testing...")
		for i, (batch, label) in enumerate(self.test_data_loader):
			  batch,label = Variable(batch.cuda()),Variable(label.cuda())              #Tensor->Variable
			  output = self.model(batch)
			  #pred = output.data.max(1)[1]
			  #correct += pred.cpu().eq(label).sum()
			  cor1, cor5 = accuracy(output.data, label, topk=(1, 5))                   # measure accuracy top1 and top5
			  correct1 += cor1
			  correct5 += cor5
			  total += label.size(0)

		if flag == -1:
		    self.accuracys1.append(float(correct1) / total)
		    self.accuracys5.append(float(correct5) / total)

		print("learningrate", self.learningrate)
		print("Accuracy Top1:", float(correct1) / total)
		print("Accuracy Top5:", float(correct5) / total)

		self.model.train()                                                              

		return float(correct1) / total, float(correct5) / total

	def adjust_learning_rate(self, epoch):
        #manually
		if self.args.learning_rate_decay == 0:
		    if epoch in [14, 21]:
		    #if epoch in [7, 14]:
		    #if epoch in []:
		        self.learningrate = self.learningrate/10;
        #exponentially
		elif self.args.learning_rate_decay == 1:
		    num_epochs = 100
		    lr_start = 0.1
		    #print("lr_start = "+str(self.lr_start))
		    lr_fin = 0.001
		    #print("lr_fin = "+str(self.lr_fin))
		    lr_decay = (lr_fin/lr_start)**(1./num_epochs)
		    #print("lr_decay = "+str(self.lr_decay))

		    self.learningrate = self.learningrate * lr_decay

def accuracy(output, target, topk=(1,)):                                               
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)                                          
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))                               

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)                                 
        res.append(correct_k)
    return res

##############################################################################################################
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--arch', '--a', default='ResNet', help='model architecture: (default: ResNet)')
    parser.add_argument('--epochs', type=int, default=28, help='number of total epochs to run')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.001, help = 'initial learning rate')
    parser.add_argument('--learning_rate_decay', '--lr_decay', type=int, default=0, help = 'maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--momentum', '--mm', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=0.0005, help='weight decay (default: 1e-4)')
    #parser.add_argument('--group_lasso_lambda', '--group_lasso_lambda', type=float, default=1e-4, help = 'group lasso lambda')
    parser.add_argument('--group_lasso_lambda', '--group_lasso_lambda', type=float, default=0, help = 'group lasso lambda')
    parser.add_argument('--print_freq', '--p', type=int, default=20, help = 'print frequency (default:20)')
    parser.add_argument('--train_path',type=str, default='/data1/Datasets/CUB200/', help = 'train dataset path')
    parser.add_argument('--test_path', type=str, default='/data1/Datasets/CUB200/', help = 'test dataset path')
    parser.set_defaults(train=True)
    args = parser.parse_args()

    return args

##############################################################################################################
if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    args = get_args()
    print("args:", args)

    model = torch.load("model_training_m40").cuda()
    model = model.module
    model.fc = nn.Linear(model.fc.in_features, 200)
    print("resnet:", model)

    #model = torch.nn.DataParallel(model).cuda()

    fine_tuner = GroupingFineTuner_ResNet(args.train_path, args.test_path, model)

    if args.train:
        fine_tuner.train(epoches = args.epochs)
        torch.save(model, "model_training_final")
        print("model_training_final:", model)
