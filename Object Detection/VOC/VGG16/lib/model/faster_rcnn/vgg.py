from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.model.utils.config import cfg
from lib.model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb
from collections import OrderedDict

from lib.model.faster_rcnn.layers import MaskedConv2d
from lib.model.faster_rcnn.layers import MaskedLinear

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()

        self.features = features
        self.classifier = nn.Sequential(
            #nn.Linear(512 * 7 * 7, 4096),
            MaskedLinear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            MaskedLinear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, num_classes),
            MaskedLinear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def set_conv_mask(self, layer_index, layer_item):
        convlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                if convlayers == layer_index:
                    for i in layer_item:
                        #print(module._mask.size())
                        module._mask[i,:,:,:] = 0
                        #print(module._mask[i,j,:,:])
                if convlayers == layer_index + 1:
                    for j in layer_item:
                        #print(module._mask.size())
                        module._mask[:,j,:,:] = 0
                        #print(module._mask[i,j,:,:])
                convlayers = convlayers + 1

    def set_linear_mask(self, layer_index, layer_item):
        linearlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedLinear'):  #[cout, cin]
                if linearlayers == layer_index:
                    for i in layer_item:
                        #print(module._mask[i,j])
                        module._mask[i,:] = 0
                        #print(module._mask[i,j])
                if linearlayers == layer_index + 1:
                    for j in layer_item:
                        #print(module._mask[i,j])
                        module._mask[:,j] = 0
                        #print(module._mask[i,j])
                linearlayers = linearlayers + 1

    def set_conv_linear_mask(self, conv_layer_index, linear_layer_index, conv_layer_item, fc_layer_item):
        convlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedConv2d'):  #[cout, cin, k, k]
                if convlayers == conv_layer_index:
                    for i in conv_layer_item:
                        #print(module._mask[i,j,:,:])
                        module._mask[i,:,:,:] = 0
                        #print(module._mask[i,j,:,:])
                convlayers = convlayers + 1

        linearlayers = 0
        for module in self.modules():
            if module.__str__().startswith('MaskedLinear'):  #[cout, cin]
                if linearlayers ==  linear_layer_index:
                    for j in fc_layer_item:
                        #print(module._mask[i,j])
                        module._mask[:,j] = 0
                        #print(module._mask[i,j])
                linearlayers = linearlayers + 1


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if i == 0:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

            else:
                conv2d = MaskedConv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)


class vgg(_fasterRCNN):
  def __init__(self, classes, pretrained=False, class_agnostic=False):
    #self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.model_path = '/data/qingbeiguo/work/faster-rcnn.pytorch-pytorch-1.0/faster-rcnn.pytorch-pytorch-1.0-1-vgg16-voc-600-20/lib/model/faster_rcnn/pretrained_model/model_training_m20.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    vgg = vgg16()

    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
        print("vgg16:", vgg)

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

    # not using the last maxpool layer
    self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

    # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

    self.RCNN_top = vgg.classifier

    # not using the last maxpool layer
    self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(4096, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)      

  def _head_to_tail(self, pool5):
    
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.RCNN_top(pool5_flat)

    return fc7

