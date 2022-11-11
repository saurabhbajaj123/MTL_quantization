import torch
import torch.nn as nn
import os
from tqdm import tqdm
import numpy as np
import time


# from post_training_quantization import MTLModel
from fusing_layers import *
from TreeMTL.data.nyuv2_dataloader_adashare import NYU_v2
from TreeMTL.data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from TreeMTL.data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics
from TreeMTL.main.trainer import Trainer
from TreeMTL.main.head import ASPPHeadNode, Classification_Module
import copy

from torch.utils.data import DataLoader


class MTLModel(nn.Module):
    def __init__(self, feature_dim=512, clsNum={}, backbone_init=None):
        super(MTLModel, self).__init__()
        self.backbone = backbone_init
        self.heads = nn.ModuleDict()
        # clsNum will contian the name of the task and the number of classes in that task
        for task in clsNum:
            self.heads[task] = ASPPHeadNode(feature_dim, clsNum[task])
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    def forward(self, x):
        x = self.quant(x)
        features = self.backbone(x)
        output = {}
        idx = 0
        for task in self.heads:
            output[task] = self.heads[task](features) # why use idx?
            output[task] = self.dequant(output[task])
        # idx += 1
        return output

task_list = [
    ['segment_semantic'],
['normal'],
['depth_zbuffer'],
['segment_semantic', 'normal'],
['segment_semantic', 'depth_zbuffer'],
['depth_zbuffer', 'normal'],
['segment_semantic','normal','depth_zbuffer']
]


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (MB):', size/1e6)
    os.remove('temp.p')
    return size

def model_compression(model_fp32, model_quantized):
    # compare the sizes
    f=print_size_of_model(model_fp32,"fp32")
    q=print_size_of_model(model_quantized,"int8")
    print("{0:.6f} times smaller".format(f/q))

def load_model(savePath, reload, clsNum):
    pretrained = resnet18_quantizable(pretrained=True)
    pretrained_features = torch.nn.Sequential(*list(pretrained.children())[:-2])
    model = MTLModel(512, clsNum, pretrained_features)

    state = torch.load(savePath + reload)
    it = state['iter']
    model.load_state_dict(state['state_dict'])

    return model, it

def validate(model, val_dataloader, tasks, criterion_dict, metric_dict, it):
    model.eval()
    loss_list = {}
    total_time, length  = 0, 0
    for task in tasks:
        loss_list[task] = []
    print("validating on {} samples = ".format(len(val_dataloader)))
    for i, data in tqdm(enumerate(val_dataloader)):
        input = data['input'] #.cuda()
        start = time.time()
        output = model(input)
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
    print("inference time = {}".format(inference_time))

    with open("validation_results.txt", 'a+') as f:
        f.write("inference time = {}".format(inference_time))
    return inference_time

def print_speedup(model_fp32, model_quantized, val_dataloader, tasks, criterion_dict, metric_dict, it):
    print('prequantization')
    inf_fp32 = validate(model_fp32, valDataloader, tasks, criterionDict, metricDict, it)
    print('postquantization')
    inf_int8 = validate(model_quantized, valDataloader, tasks, criterionDict, metricDict, it)
    print("Speed Up = {}".format(inf_fp32/inf_int8))

dataroot = "/work/sbajaj_umass_edu/Datasets/nyu_v2"
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
trainDataloader = DataLoader(dataset, 16, shuffle=True)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
valDataloader = DataLoader(dataset, 16, shuffle=True)

for tasks in task_list:

    clsNum = {}
    criterionDict = {}
    metricDict = {}
    task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}
    for task in tasks:
        criterionDict[task] = NYUCriterions(task)
        metricDict[task] = NYUMetrics(task)
        clsNum[task] = task_cls_num[task]

    savePath = "/work/sbajaj_umass_edu/BestModels/"
    reload = "_".join(tasks) + '.model'

    model, it = load_model(savePath, reload, clsNum)
    device = torch.device('cpu')
    model_fp32 = model.to(device)

    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False)
    model_quantized = torch.quantization.convert(model_fp32_prepared)

    model_compression(model_fp32, model_quantized)
    print_speedup(model_fp32, model_quantized, valDataloader, tasks, criterionDict, metricDict, it)