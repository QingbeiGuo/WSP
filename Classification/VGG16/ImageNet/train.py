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
#model
from models.vgg.vgg import VGG, vgg16
#dataset
import dataset.imagenet.dataset_imagenet

##############################################################################################################
class FineTuner_CNN:
	def __init__(self, train_path, test_path, model):
	    self.args = args
	    self.learningrate = self.args.learning_rate
	    self.learning_rate_decay = self.args.learning_rate_decay
	    self.momentum = self.args.momentum
	    self.weight_decay = self.args.weight_decay
	    self.conv_compress_percentage = self.args.conv_compress_percentage
	    self.fc_compress_percentage = self.args.fc_compress_percentage
	    self.conv_fc_compress_percentage = self.args.conv_fc_compress_percentage
	    self.train_path = self.args.train_path
	    self.test_path = self.args.test_path

#	    #imagenet
	    self.train_data_loader = dataset.imagenet.dataset_imagenet.train_loader(self.train_path)
	    self.test_data_loader  = dataset.imagenet.dataset_imagenet.test_loader(self.test_path)

	    self.model = model
	    self.criterion = torch.nn.CrossEntropyLoss()

	    self.accuracys1 = []
	    self.accuracys5 = []

	    self.L1_conv_channels = []
	    self.L1_conv_filters = []
	    self.L1_fc_channels = []
	    self.L1_fc_filters = []

	    self.L1_conv_pair = []
	    self.L1_fc_pair = []

	    self.conv_candidate_idx = []
	    self.fc_candidate_idx = []

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
		    #imagenet
		    if epoch in [30, 60]:
		        self.learningrate = self.learningrate/10;
        #exponentially
		elif self.args.learning_rate_decay == 1:
		    num_epochs = 60
		    lr_start = 0.01
		    #print("lr_start = "+str(self.lr_start))
		    lr_fin = 0.0001
		    #print("lr_fin = "+str(self.lr_fin))
		    lr_decay = (lr_fin/lr_start)**(1./num_epochs)
		    #print("lr_decay = "+str(self.lr_decay))

		    self.learningrate = self.learningrate * lr_decay

##############################################################################################################
	def compress(self):
		fine_tuner.calcaulate()
		fine_tuner.pair()
		fine_tuner.candidate()
		fine_tuner.prune()

	def calcaulate(self):
        #L1
		self.L1_conv_channels = []
		self.L1_conv_filters = []
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedConv2d'):
		        #print("weight size:", module.weight.size()) #[cout, cin, k, k]
		        #weight = abs(module.weight)
		        weight = abs(module.weight * module._mask)
		        #weight = module.weight.pow(2) * module._mask

		        L1_channel = weight.sum(3).sum(2).sum(0)
		        L1_filter = weight.sum(3).sum(2).sum(1)

		        self.L1_conv_channels.append(L1_channel.cpu().data)
		        self.L1_conv_filters.append(L1_filter.cpu().data)
		print("L1_conv_channels:",self.L1_conv_channels)
		print("L1_conv_filters:",self.L1_conv_filters)

		self.L1_fc_channels = []
		self.L1_fc_filters = []
		for module in self.model.modules():
		    if module.__str__().startswith('MaskedLinear'):
		        #print("weight size:", module.weight.size()) #[cout, cin]
		        #weight = abs(module.weight)
		        weight = abs(module.weight * module._mask)
		        #weight = module.weight.pow(2) * module._mask

		        L1_channel = weight.sum(0)
		        L1_filter = weight.sum(1)

		        self.L1_fc_channels.append(L1_channel.cpu().data)
		        self.L1_fc_filters.append(L1_filter.cpu().data)
		print("L1_fc_channels:",self.L1_fc_channels)
		print("L1_fc_filters:",self.L1_fc_filters)

	def pair(self):
        #pair on L1 of channel and filter
		self.L1_conv_pair = []
		for i in range(len(self.L1_conv_filters)-1):
		    self.L1_conv_pair.append([self.L1_conv_filters[i],self.L1_conv_channels[i+1]])
		print("L1_conv_pair:",self.L1_conv_pair)

		self.L1_fc_pair = []
		for j in range(len(self.L1_fc_filters)-1):
		    self.L1_fc_pair.append([self.L1_fc_filters[j],self.L1_fc_channels[j+1]])
		print("L1_fc_pair:",self.L1_fc_pair)

		print("self.L1_fc_channels[0]",self.L1_fc_channels[0])
		print("self.L1_fc_channels[0].view(256, -1)",self.L1_fc_channels[0].view(256, -1))
		print("self.L1_fc_channels[0].view(256, -1).sum(1)",self.L1_fc_channels[0].view(256, -1).sum(1))
		self.L1_conv_fc_pair = []
		self.L1_conv_fc_pair.append([self.L1_conv_filters[len(self.L1_conv_filters)-1],self.L1_fc_channels[0].view(256, -1).sum(1)])
		print("L1_conv_fc_pair:",self.L1_conv_fc_pair)

	def candidate(self):
        #based on L1 of channel and filter
		self.conv_candidate_idx = []
		for i in range(len(self.L1_conv_pair)):
		    threshold_conv_filter = np.percentile(self.L1_conv_pair[i][0], self.conv_compress_percentage) #L1_conv_filters
		    print("threshold_conv_filter", threshold_conv_filter)
		    candidate_filter = np.where(np.array(self.L1_conv_pair[i][0]) < threshold_conv_filter)[0]
		    print("candidate_filter", candidate_filter)

		    threshold_conv_channel = np.percentile(self.L1_conv_pair[i][1], self.conv_compress_percentage) #L1_conv_channels
		    print("threshold_conv_channel", threshold_conv_channel)
		    candidate_channel = np.where(np.array(self.L1_conv_pair[i][1]) < threshold_conv_channel)[0]
		    print("candidate_channel", candidate_channel)

		    conv_candidate_idx = list(set(candidate_filter) & set(candidate_channel))
		    print("conv_candidate_idx", conv_candidate_idx)
		    self.conv_candidate_idx.append(conv_candidate_idx)
		    print("self.conv_candidate_idx", self.conv_candidate_idx)

		self.fc_candidate_idx = []
		for i in range(len(self.L1_fc_pair)):
		    threshold_fc_filter = np.percentile(self.L1_fc_pair[i][0], self.fc_compress_percentage) #L1_fc_filters
		    print("threshold_fc_filter", threshold_fc_filter)
		    candidate_filter = np.where(np.array(self.L1_fc_pair[i][0]) < threshold_fc_filter)[0]
		    print("candidate_filter", candidate_filter)

		    threshold_fc_channel = np.percentile(self.L1_fc_pair[i][1], self.fc_compress_percentage) #L1_fc_channels
		    print("threshold_fc_channel", threshold_fc_channel)
		    candidate_channel = np.where(np.array(self.L1_fc_pair[i][1]) < threshold_fc_channel)[0]
		    print("candidate_channel", candidate_channel)

		    fc_candidate_idx = list(set(candidate_filter) & set(candidate_channel))
		    print("fc_candidate_idx", fc_candidate_idx)
		    self.fc_candidate_idx.append(fc_candidate_idx)
		    print("self.fc_candidate_idx", self.fc_candidate_idx)

		self.conv_fc_candidate_idx = []
		for i in range(len(self.L1_conv_fc_pair)):
		    threshold_conv_fc_filter = np.percentile(self.L1_conv_fc_pair[i][0], self.conv_fc_compress_percentage) #L1_conv_fc_filters
		    print("threshold_conv_fc_filter", threshold_conv_fc_filter)
		    candidate_filter = np.where(np.array(self.L1_conv_fc_pair[i][0]) < threshold_conv_fc_filter)[0]
		    print("candidate_filter", candidate_filter)

		    threshold_conv_fc_channel = np.percentile(self.L1_conv_fc_pair[i][1], self.conv_fc_compress_percentage) #L1_conv_fc_channels
		    print("threshold_conv_fc_channel", threshold_conv_fc_channel)
		    candidate_channel = np.where(np.array(self.L1_conv_fc_pair[i][1]) < threshold_conv_fc_channel)[0]
		    print("candidate_channel", candidate_channel)

		    conv_fc_candidate_idx = list(set(candidate_filter) & set(candidate_channel))
		    print("conv_fc_candidate_idx", conv_fc_candidate_idx)
		    self.conv_fc_candidate_idx.append(conv_fc_candidate_idx)
		    print("self.conv_fc_candidate_idx", self.conv_fc_candidate_idx)

	def prune(self):
		#pruned based on ICP scheme
		for i, conv_item in enumerate(list(self.conv_candidate_idx)):
		    print("i-conv", i)
		    print("item-conv", conv_item)
		    #mask pruning
		    self.model.set_conv_mask(i, conv_item)

		for j, fc_item in enumerate(list(self.fc_candidate_idx)):
		    print("i-fc", j)
		    print("item-fc", fc_item)
		    #mask pruning
		    self.model.set_linear_mask(j, fc_item)

		for k, conv_item in enumerate(list(self.conv_fc_candidate_idx)):
		    print("i-conv_fc", k)
		    print("item-conv_fc1", conv_item)
		    fc_item = []
		    for l in conv_item:
		        fc_item = fc_item + list(range(l*49, (l+1)*49))
		    print("item-conv_fc2", fc_item)
		    #mask pruning
		    self.model.set_conv_linear_mask(i+1, 0, conv_item, fc_item)

#	def normalization(self, vector):
#		return vector / vector.sum()

##############################################################################################################

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
    parser = argparse.ArgumentParser(description='PyTorch CNN Training')

    parser.add_argument('--arch', '--a', default='AlexNet', help='model architecture: (default: AlexNet)')
    parser.add_argument('--epochs', type=int, default=90, help='number of total epochs to run')
    parser.add_argument('--learning_rate', '--lr', type=float, default=0.01, help = 'initial learning rate')
    parser.add_argument('--learning_rate_decay', '--lr_decay', type=int, default=0, help = 'maually[0] or exponentially[1] decaying learning rate')
    parser.add_argument('--momentum', '--mm', type=float, default=0.9, help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', '--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
#    parser.add_argument('--weight_decay', '--wd', type=float, default=0, help='weight decay (default: 1e-4)')
    parser.add_argument('--conv_compress_percentage', '--conv_compress_percentage', type=int, default=20, help = 'compress percentage')
    parser.add_argument('--fc_compress_percentage', '--fc_compress_percentage', type=int, default=20, help = 'compress percentage')
    parser.add_argument('--conv_fc_compress_percentage', '--conv_fc_compress_percentage', type=int, default=20, help = 'compress percentage')
    parser.add_argument('--print_freq', '--p', type=int, default=100, help = 'print frequency (default:20)')
#    #imagenet
    parser.add_argument('--train_path',type=str, default='/data1/Datasets/ImageNet/ILSVRC2012/ILSVRC2012_img_train/', help = 'train dataset path')
    parser.add_argument('--test_path', type=str, default='/data1/Datasets/ImageNet/ILSVRC2012/ILSVRC2012_img_val_subfolders/', help = 'test dataset path')
    parser.add_argument("--parallel", type = int, default = 1)
    parser.set_defaults(compress=False)
    parser.set_defaults(train=True)
    args = parser.parse_args()

    return args

##############################################################################################################
if __name__ == '__main__':
#	os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	args = get_args()
	print("args:", args)

	'''model = torch.load("model_training_m").module
	print("model_training:", model)
	torch.save(model.state_dict(), "model_training_m20.pth")
	print("model_training:", model.state_dict())'''

	'''vgg = models.vgg16(pretrained=True)
	print("vgg:", vgg)
	model = vgg16(pretrained=False)
	print("vgg16:", model)

	pretrained_dict = vgg.state_dict()
	#print("pretrained_dict", pretrained_dict)
	model_dict = model.state_dict()

	pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}
	model_dict.update(pretrained_dict)
	model.load_state_dict(model_dict)
	#print("model_dict", model_dict)
	#print("model_training:", model)
	torch.save(model, "model")'''

	#model = models.vgg16(pretrained=True).cuda()
	model = torch.load("model_compress_20").cuda()
	print("model_training:", model)

	if args.parallel == 1:
	    model = torch.nn.DataParallel(model).cuda()

	fine_tuner = FineTuner_CNN(args.train_path, args.test_path, model)
	#fine_tuner.test(0)

	if args.compress:
	    fine_tuner.compress()
	    torch.save(model, "model_compress_20")
	elif args.train:
	    fine_tuner.train(epoches = args.epochs)
	    torch.save(model, "model_training_final")
	    print("model_training_final:", model)