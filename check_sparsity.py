#%%
import pwd
import numpy as np
import os
import copy
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
import argparse
from pathlib import Path
from copy import deepcopy
import torchvision.models as models
import torchvision
from torch.utils.tensorboard import SummaryWriter
from DGMSParent.utils.sparsity import SparsityMeasure


# from tensorboard import program


from TreeMTL.data.nyuv2_dataloader_adashare import NYU_v2
from TreeMTL.data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from TreeMTL.data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics
from TreeMTL.main.trainer import Trainer
from TreeMTL.main.head import ASPPHeadNode, Classification_Module

# from DGMS.utils.sparsity import SparsityMeasure
# from DGMS.modeling import DGMSNet
from DGMSParent.utils.PyTransformer.transformers.torchTransformer import TorchTransformer



tasks = ('segment_semantic','normal','depth_zbuffer')
task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

criterionDict = {}
metricDict = {}
clsNum = {}
three_task = ['segment_semantic','normal','depth_zbuffer']

dataroot = "/home/sbajaj/MyCode/quantization_remote/Datasets/nyu_v2"
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
trainDataloader = DataLoader(dataset, 16, shuffle=True)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
valDataloader = DataLoader(dataset, 16, shuffle=True)
new_model = False

for task in three_task:
    criterionDict[task] = NYUCriterions(task)
    metricDict[task] = NYUMetrics(task)
    clsNum[task] = task_cls_num[task]
# print(criterionDict, metricDict, clsNum)

# print(len(valDataloader), len(trainDataloader))


# Importing a pre-trained resnet

# pretrained = models.resnet34(pretrained=True)
pretrained = models.resnet18(pretrained=True)

### strip the last layer
pretrained_features = torch.nn.Sequential(*list(pretrained.children())[:-2])
### check this works
# x = torch.randn([1,3,224,224])
# features = pretrained_features(x) # output now has the features corresponding to input x
# print(pretrained)
# print(pretrained_features)
# print(pretrained_features[4][0].conv1.weight)

# Defining the quantization module
# from __future__ import print_function

# import torch.nn as nn

# from .networks import get_network
# from DGMS.modeling.DGMS import DGMSConv
# os.chdir("/home/sbajaj/MyCode/quantization_remote/MTL_quantization/DGMSParent")
# print()
# print(os.getcwd(),  os.listdir('.'))
from DGMSParent.modeling.DGMS import DGMSConv
# os.chdir("/home/sbajaj/MyCode/quantization_remote/MTL_quantization")
# print(os.getcwd())

import torchvision.models as models

class DGMSNet(nn.Module):
    def __init__(self, network, freeze_weights=False, freeze_bn=False):
        super(DGMSNet, self).__init__()
        # self.args = args
        # self.args.freeze_weights = False
        self.freeze_weights = freeze_weights
        self.network = network # models.__dict__['resnet34'](pretrained=True) # get_network(args)
        self.freeze_bn = freeze_bn

    def init_mask_params(self):
        print("--> Start to initialize sub-distribution parameters, this may take some time...")
        for name, m in self.network.named_modules():
            # if isinstance(m, nn.Conv2d):
            #     m = DGMSConv(m.in_channels, m.out_channels, m.kernel_size, m.stride, m.padding, m.groups)
            if isinstance(m, DGMSConv):
                m.init_mask_params()
        print("--> Sub-distribution parameters initialization finished!")

    def forward(self, x):
        x = self.network(x)
        return x

    def get_1x_lr_params(self):
        self.init_mask_params()
        modules = [self.network]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        # if self.args.freeze_weights:
                        if self.freeze_weights:
                            for p in m[1].parameters():
                                pass
                        else:
                            for p in m[1].parameters():
                                if p.requires_grad:
                                    yield p
                else:
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p




class MTLModel(nn.Module):
  def __init__(self, feature_dim=512, clsNum={}, backbone_init=None):
    super(MTLModel, self).__init__()
    self.backbone = backbone_init
    self.heads = nn.ModuleDict()
    # clsNum will contian the name of the task and the number of classes in that task
    for task in clsNum:
      self.heads[task] = ASPPHeadNode(feature_dim, clsNum[task])
  
  def forward(self, x):
    features = self.backbone(x)
    output = {}
    idx = 0
    for task in self.heads:
      output[task] = self.heads[task](features) # why use idx?
      # idx += 1
    return output

# model = MTLModel(512, clsNum, model)
model = MTLModel(512, clsNum, pretrained_features)

# model = DGMSNet(pretrained_features)
model = DGMSNet(model)

_transformer = TorchTransformer()
_transformer.register(nn.Conv2d, DGMSConv)
model = _transformer.trans_layers(model)
sparsity = SparsityMeasure(None)

timings = defaultdict(list)
evaluation_time = defaultdict(list)
for i in range(10):
    for pow in range(-2, -8, -1):
        loss_list = {}
        for task in tasks:
            loss_list[task] = []
        print("loading_model = {}".format(str(pow)))
        t1_load = time.time()
        model = torch.load("/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/resnet18_DGMS_MTL_full_trained_K3_1e-{}_test2_iters_199.pt".format(str(pow)))
        t2_load = time.time()
        model = model.cuda()
        model.eval()
        eval_time = 0
        for i, data in enumerate(valDataloader):
            x = data['input'].cuda()

            t1_eval = time.time()
            output = model(x)
            t2_eval = time.time()
            eval_time += t2_eval - t1_eval
            # for task in tasks:
            #     y = data[task].cuda()
            #     if task + '_mask' in data:
            #         tloss = criterionDict[task](output[task], y, data[task + '_mask'].cuda())
            #         metricDict[task](output[task], y, data[task + '_mask'].cuda())
            #     else:
            #         tloss = criterionDict[task](output[task], y)
            #         metricDict[task](output[task], y)
            #     loss_list[task].append(tloss.item())
        # total_sparse_ratio, model_params, compression_rate = sparsity.check_sparsity_per_layer(model)
        if i > 1: 
            timings[pow].append(t2_load - t1_load)
            evaluation_time[pow].append(eval_time)

for pow in range(-2, -8, -1):
    with open("output1.txt", 'a+') as f:
        f.write('\n')
        # f.write("resnet18_DGMS_MTL_full_trained_K3_1e-{}_test2_iters_199.pt, compression = {} \n".format(str(pow), compression_rate))
        f.write("K3_1e-{}: time to load avg = {}, avg evalTime = {}".format(str(pow), np.mean(timings[pow]), np.mean(evaluation_time[pow])))
#%%
plt.figure(1)
for key, data_list in timings.items():
    plt.plot(data_list, label=key)
plt.legend()
plt.savefig('load_timings.png')


plt.figure(2)
for key, data_list in evaluation_time.items():
    plt.plot(data_list, label=key)
plt.legend()
plt.savefig('evaluation_time.png')