
import pwd
import numpy as np
import os
import copy
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy
import torchvision.models as models
import torchvision
from torch.utils.tensorboard import SummaryWriter

from TreeMTL.data.nyuv2_dataloader_adashare import NYU_v2
from TreeMTL.data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from TreeMTL.data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics
from TreeMTL.main.trainer import Trainer
from TreeMTL.main.head import ASPPHeadNode, Classification_Module

from fusing_layers import *

new_model = True
perform_quantization = False

tasks = ('segment_semantic','normal','depth_zbuffer')
task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
three_task = ['depth_zbuffer']
criterionDict = {}
metricDict = {}
clsNum = {}

dataroot = "/home/sbajaj/MyCode/quantization_remote/Datasets/nyu_v2"
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
trainDataloader = DataLoader(dataset, 16, shuffle=True)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
valDataloader = DataLoader(dataset, 16, shuffle=True)

for task in three_task:
    criterionDict[task] = NYUCriterions(task)
    metricDict[task] = NYUMetrics(task)
    clsNum[task] = task_cls_num[task]
print(criterionDict, metricDict, clsNum)

print(len(valDataloader), len(trainDataloader))

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
# pretrained = models.resnet18(pretrained=True)
pretrained = resnet18_quantizable(pretrained=True)
print('quantizable model loaded')
# print("pretrained == {}".format(pretrained))
pretrained_features = torch.nn.Sequential(*(list(pretrained.children())[:-4] + list(pretrained.children())[-2:]))
# print("pretrained_features == {}".format(pretrained_features))
model = MTLModel(512, clsNum, pretrained_features)
print(model)
# savePath = '/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/trained'
# state = torch.load(savePath + '_' + '_'.join(three_task) + '.model')
# model.load_state_dict(state['state_dict'])
# startIter = state['iter'] + 1

# model = model.cuda()
model = model.to('cpu')

def validate( model, val_dataloader, tasks, criterion_dict, metric_dict, it):
    model.eval()
    loss_list = {}
    total_time, length  = 0, 0
    for task in tasks:
        loss_list[task] = []
    print("validating on {} samples = ".format(len(val_dataloader)))
    for i, data in enumerate(val_dataloader):
        x = data['input'] #.cuda()
        start = time.time()
        output = model(x)
        end = time.time()
        delta = end - start
        total_time += delta
        length += len(data)
        for task in tasks:
            y = data[task]#.cuda()
            if task + '_mask' in data:
                # tloss = criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
                # metric_dict[task](output[task], y, data[task + '_mask'].cuda())
                tloss = criterion_dict[task](output[task], y, data[task + '_mask'])
                metric_dict[task](output[task], y, data[task + '_mask'])
            else:
                tloss = criterion_dict[task](output[task], y)
                metric_dict[task](output[task], y)
            loss_list[task].append(tloss.item())

    for task in tasks:
        val_results = metric_dict[task].val_metrics()
        print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])), flush=True)
        print(val_results, flush=True)
        with open("validation_results.txt", 'a+') as f:
            f.write('\n')
            f.write('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])))
            f.write('\n')
            f.write(str(val_results))
            f.write('\n')
    print('======================================================================', flush=True)


    inference_time = total_time/length
    with open("validation_results.txt", 'a+') as f:
        f.write("inference time = {}".format(inference_time))
    return

# validate(model, valDataloader, three_task, criterionDict, metricDict, startIter)

model_fp32 = model
model_fp32.eval()

model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# print(model_fp32)
model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False)

startIter = 1
print('prequantization')
validate(model_fp32_prepared, valDataloader, three_task, criterionDict, metricDict, startIter)
model_quantized = torch.quantization.convert(model_fp32_prepared) 
print('postquantization')
validate(model_quantized, valDataloader, three_task, criterionDict, metricDict, startIter)