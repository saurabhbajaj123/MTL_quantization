
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
# three_task = ['segment_semantic']
# three_task = ['normal']
# three_task = ['depth_zbuffer']
# three_task = ['segment_semantic', 'normal']
# three_task = ['segment_semantic', 'depth_zbuffer']
three_task = ['depth_zbuffer', 'normal']
# three_task = ['segment_semantic','normal','depth_zbuffer']

# three_task = [
#     ['segment_semantic'],
#     ['normal'],
#     ['depth_zbuffer'],
#     ['segment_semantic', 'normal'],
#     ['segment_semantic', 'depth_zbuffer'],
#     ['depth_zbuffer', 'normal'],
#     ['segment_semantic','normal','depth_zbuffer'],
# ]

criterionDict = {}
metricDict = {}
clsNum = {}

dataroot = "/work/sbajaj_umass_edu/Datasets/nyu_v2"
dataset = NYU_v2(dataroot, 'train', crop_h=321, crop_w=321)
trainDataloader = DataLoader(dataset, 16, shuffle=True)

dataset = NYU_v2(dataroot, 'test', crop_h=321, crop_w=321)
valDataloader = DataLoader(dataset, 16, shuffle=True)

for task in three_task:
    criterionDict[task] = NYUCriterions(task)
    metricDict[task] = NYUMetrics(task)
    clsNum[task] = task_cls_num[task]
# print(criterionDict, metricDict, clsNum)

print(len(valDataloader), len(trainDataloader))

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
# pretrained = models.resnet18(pretrained=True)
pretrained = resnet18_quantizable(pretrained=True)
print('quantizable model loaded')
# print("pretrained == {}".format(pretrained))
pretrained_features = torch.nn.Sequential(*list(pretrained.children())[:-2])
# pretrained_features = torch.nn.Sequential(*(list(pretrained.children())[:-4] + list(pretrained.children())[-2:]))
# print("pretrained_features == {}".format(pretrained_features))
model = MTLModel(512, clsNum, pretrained_features)
# print(model)

# savePath = '/work/sbajaj_umass_edu/SavedModels/'
# num_iters = 3
# # state = torch.load(savePath + 'resnet18' +  '_' + '_'.join(three_task) + '_iters_'+ str(num_iters) + '.pt')
# state = torch.load(savePath + "".join(three_task))
# print(state.keys())
# model.load_state_dict(state)

# startIter = state['iter'] + 1


def validate( model, val_dataloader, tasks, criterion_dict, metric_dict, it):
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
    return

loss_lambda = {'segment_semantic': 1, 'normal':1, 'depth_zbuffer': 1}
checkpoint = 'Checkpoints/NYUv2/test/'
tb = SummaryWriter()
class Trainer():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict, 
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 print_iters=50, val_iters=200, save_iters=200, optimizer=None, quantization=False, startIter=0, bestPath=None):
        print("initializing the Trainer")
        super(Trainer, self).__init__()
        self.model = model
        self.startIter = startIter
        self.quantization = quantization
        if not optimizer:
          self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
        else: 
          self.optimizer = optimizer
        
                
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=decay_lr_freq, gamma=decay_lr_rate)
        
        self.tasks = tasks
        
        self.train_dataloader = train_dataloader
        self.train_iter = iter(self.train_dataloader)
        self.val_dataloader = val_dataloader
        self.criterion_dict = criterion_dict
        self.metric_dict = metric_dict
        
        self.loss_list = {}
        self.set_train_loss()
        
        self.print_iters = print_iters
        self.val_iters = val_iters
        self.save_iters = save_iters
        self.tb_data = next(iter(trainDataloader))

        self.best_val_loss = float("inf")


        if bestPath:
            state = torch.load(bestPath)
            self.model.load_state_dict(state['state_dict'])
            self.model.eval()
            loss_list = {}
            for task in self.tasks:
                loss_list[task] = []

            for i, data in enumerate(self.val_dataloader):
                x = data['input'].cuda()
                # x = data['input']

                output = self.model(x)
                
                for task in self.tasks:
                    y = data[task].cuda()
                    # y = data[task]
                    
                    if task + '_mask' in data:
                        tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
                        self.metric_dict[task](output[task], y, data[task + '_mask'].cuda())              
                    else:
                        tloss = self.criterion_dict[task](output[task], y)
                        self.metric_dict[task](output[task], y)
                    loss_list[task].append(tloss.item())
            val_loss = 0
            for task in self.tasks:
                val_loss += np.mean(loss_list[task])
            print("model with best val_loss = {}".format(val_loss))
            self.best_val_loss = val_loss
            

    def train(self, iters, loss_lambda, savePath=None, reload=None):
        self.model.train()
        if reload is not None and reload != 'false' and savePath is not None:
            self.load_model(savePath, reload)
        # tb = SummaryWriter()
        # if self.tb_data['input']:
        grid = torchvision.utils.make_grid(self.tb_data['input'])
        tb.add_image('images', grid)
        # tb.add_graph(model, self.tb_data['input'])
        

        
        for i in range(self.startIter, iters):
            # print('i = {}'.format(i))
            self.train_step(loss_lambda)

            if (i+1) % self.print_iters == 0:
                self.print_train_loss(i)
                tb.add_scalar('Loss_train', self.get_loss(i), i)
                self.set_train_loss()
            if (i+1) % self.val_iters == 0:
                self.validate(i)
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    self.save_model(i, savePath)
                    # torch.save(model, "/work/sbajaj_umass_edu/SavedModels/resnet18_{}_iters_{}.pt".format("_".join(three_task), i))
                    # torch.save(model.state_dict(), "/work/sbajaj_umass_edu/SavedModels/{}".format("".join(three_task)))
            # print(self.get_loss(i))
            # tb.add_scalar('Loss', self.get_loss(i), i)


            # ############## Quantization #####################
            # if self.quantization: 
            #     tb.add_histogram('conv1.weight', model.network.backbone[0].weight, i)
            #     tb.add_histogram('4.0.conv1.weight', model.network.backbone[4][0].conv1.weight, i)
            #     tb.add_histogram('4.0.conv2.weight', model.network.backbone[4][0].conv2.weight, i)
            #     tb.add_histogram('4.1.conv1.weight', model.network.backbone[4][1].conv1.weight, i)
            #     tb.add_histogram('4.1.conv2.weight', model.network.backbone[4][1].conv2.weight, i)
            #     # tb.add_histogram('4.2.conv1.weight', model.network.backbone[4][2].conv1.weight, i)
            #     # tb.add_histogram('4.2.conv2.weight', model.network.backbone[4][2].conv2.weight, i)
            #     tb.add_histogram('5.0.conv1.weight', model.network.backbone[5][0].conv1.weight, i)
            #     tb.add_histogram('5.0.conv2.weight', model.network.backbone[5][0].conv2.weight, i)
            #     tb.add_histogram('5.1.conv1.weight', model.network.backbone[5][1].conv1.weight, i)
            #     tb.add_histogram('5.1.conv2.weight', model.network.backbone[5][1].conv2.weight, i)
            #     # tb.add_histogram('5.2.conv1.weight', model.network.backbone[5][2].conv1.weight, i)
            #     # tb.add_histogram('5.2.conv2.weight', model.network.backbone[5][2].conv2.weight, i)
            #     # tb.add_histogram('5.3.conv1.weight', model.network.backbone[5][3].conv1.weight, i)
            #     # tb.add_histogram('5.3.conv2.weight', model.network.backbone[5][3].conv2.weight, i)
            #     tb.add_histogram('6.0.conv1.weight', model.network.backbone[6][0].conv1.weight, i)
            #     tb.add_histogram('6.0.conv2.weight', model.network.backbone[6][0].conv2.weight, i)
            #     tb.add_histogram('6.1.conv1.weight', model.network.backbone[6][1].conv1.weight, i)
            #     tb.add_histogram('6.1.conv2.weight', model.network.backbone[6][1].conv2.weight, i)
            #     # tb.add_histogram('6.2.conv1.weight', model.network.backbone[6][2].conv1.weight, i)
            #     # tb.add_histogram('6.2.conv2.weight', model.network.backbone[6][2].conv2.weight, i)
            #     # tb.add_histogram('6.3.conv1.weight', model.network.backbone[6][3].conv1.weight, i)
            #     # tb.add_histogram('6.3.conv2.weight', model.network.backbone[6][3].conv2.weight, i)
            #     # tb.add_histogram('6.4.conv1.weight', model.network.backbone[6][4].conv1.weight, i)
            #     # tb.add_histogram('6.4.conv2.weight', model.network.backbone[6][4].conv2.weight, i)
            #     # tb.add_histogram('6.5.conv1.weight', model.network.backbone[6][5].conv1.weight, i)
            #     # tb.add_histogram('6.5.conv2.weight', model.network.backbone[6][5].conv2.weight, i)
            #     tb.add_histogram('7.0.conv1.weight', model.network.backbone[7][0].conv1.weight, i)
            #     tb.add_histogram('7.0.conv2.weight', model.network.backbone[7][0].conv2.weight, i)
            #     tb.add_histogram('7.1.conv1.weight', model.network.backbone[7][1].conv1.weight, i)
            #     tb.add_histogram('7.1.conv2.weight', model.network.backbone[7][1].conv2.weight, i)
            # else:

            #     ####### Normal Execution ######
            #     tb.add_histogram('conv1.weight', model.backbone[0].weight, i)
            #     tb.add_histogram('4.0.conv1.weight', model.backbone[4][0].conv1.weight, i)
            #     tb.add_histogram('4.0.conv2.weight', model.backbone[4][0].conv2.weight, i)
            #     tb.add_histogram('4.1.conv1.weight', model.backbone[4][1].conv1.weight, i)
            #     tb.add_histogram('4.1.conv2.weight', model.backbone[4][1].conv2.weight, i)
            #     # tb.add_histogram('4.2.conv1.weight', model.backbone[4][2].conv1.weight, i)
            #     # tb.add_histogram('4.2.conv2.weight', model.backbone[4][2].conv2.weight, i)
            #     tb.add_histogram('5.0.conv1.weight', model.backbone[5][0].conv1.weight, i)
            #     tb.add_histogram('5.0.conv2.weight', model.backbone[5][0].conv2.weight, i)
            #     tb.add_histogram('5.1.conv1.weight', model.backbone[5][1].conv1.weight, i)
            #     tb.add_histogram('5.1.conv2.weight', model.backbone[5][1].conv2.weight, i)
            #     # tb.add_histogram('5.2.conv1.weight', model.backbone[5][2].conv1.weight, i)
            #     # tb.add_histogram('5.2.conv2.weight', model.backbone[5][2].conv2.weight, i)
            #     # tb.add_histogram('5.3.conv1.weight', model.backbone[5][3].conv1.weight, i)
            #     # tb.add_histogram('5.3.conv2.weight', model.backbone[5][3].conv2.weight, i)
            #     tb.add_histogram('6.0.conv1.weight', model.backbone[6][0].conv1.weight, i)
            #     tb.add_histogram('6.0.conv2.weight', model.backbone[6][0].conv2.weight, i)
            #     tb.add_histogram('6.1.conv1.weight', model.backbone[6][1].conv1.weight, i)
            #     tb.add_histogram('6.1.conv2.weight', model.backbone[6][1].conv2.weight, i)
            #     # tb.add_histogram('6.2.conv1.weight', model.backbone[6][2].conv1.weight, i)
            #     # tb.add_histogram('6.2.conv2.weight', model.backbone[6][2].conv2.weight, i)
            #     # tb.add_histogram('6.3.conv1.weight', model.backbone[6][3].conv1.weight, i)
            #     # tb.add_histogram('6.3.conv2.weight', model.backbone[6][3].conv2.weight, i)
            #     # tb.add_histogram('6.4.conv1.weight', model.backbone[6][4].conv1.weight, i)
            #     # tb.add_histogram('6.4.conv2.weight', model.backbone[6][4].conv2.weight, i)
            #     # tb.add_histogram('6.5.conv1.weight', model.backbone[6][5].conv1.weight, i)
            #     # tb.add_histogram('6.5.conv2.weight', model.backbone[6][5].conv2.weight, i)
            #     tb.add_histogram('7.0.conv1.weight', model.backbone[7][0].conv1.weight, i)
            #     tb.add_histogram('7.0.conv2.weight', model.backbone[7][0].conv2.weight, i)
            #     tb.add_histogram('7.1.conv1.weight', model.backbone[7][1].conv1.weight, i)
            #     tb.add_histogram('7.1.conv2.weight', model.backbone[7][1].conv2.weight, i)


            # tb.close()
        # Reset loss list and the data iters
        self.set_train_loss()
        return
    
    def train_step(self, loss_lambda):
        # print("starting train step")
        self.model.train()


        try:
            data = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_dataloader)
            data = next(self.train_iter)
        
        x = data['input'].cuda()
        # x = data['input']
        # print("size(x) = ", x.size())
        self.optimizer.zero_grad()
        output = self.model(x)
        # print("output = {}".format(output.size()))
        loss = 0
        for task in self.tasks:
            # print("task = {}".format(task))
            y = data[task].cuda()
            # y = data[task]

            if task + '_mask' in data:
                tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
                # tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'])
            else:
                tloss = self.criterion_dict[task](output[task], y)
                
            self.loss_list[task].append(tloss.item())
            loss += loss_lambda[task] * tloss
            # print("loss = {}".format(loss))
        self.loss_list['total'].append(loss.item())
        # print(self.loss_list['total'])
        
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        return
    
    def validate(self, it):
        self.model.eval()
        loss_list = {}
        for task in self.tasks:
            loss_list[task] = []
        print("validating on {} samples = ".format(len(self.val_dataloader)))
        for i, data in enumerate(self.val_dataloader):
            x = data['input'].cuda()
            # x = data['input']

            output = self.model(x)
            
            for task in self.tasks:
                y = data[task].cuda()
                # y = data[task]
                
                if task + '_mask' in data:
                    tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
                    self.metric_dict[task](output[task], y, data[task + '_mask'].cuda())
                    # tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'])
                    # self.metric_dict[task](output[task], y, data[task + '_mask'])                
                else:
                    tloss = self.criterion_dict[task](output[task], y)
                    self.metric_dict[task](output[task], y)
                loss_list[task].append(tloss.item())
        # with open("validation_results.txt", 'a+') as f:
            # f.write('______________________________________________________________________')
            # f.write("{}_{}.pt".format("resnet18_DGMS_MTL_full", "K3_1e--2_test2"))

        for task in self.tasks:
            val_results = self.metric_dict[task].val_metrics()
            print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])), flush=True)
            tb.add_scalar(str(task[:4]) + "_val", np.mean(loss_list[task]), it)
            print(val_results, flush=True)
            # with open("validation_results.txt", 'a+') as f:
            #     f.write('\n')
            #     f.write('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])))
            #     f.write('\n')
            #     f.write(str(val_results))
            #     f.write('\n')
        val_loss = 0
        for task in self.tasks:
            val_loss += np.mean(loss_list[task])
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_model(it, "/work/sbajaj_umass_edu/BestModels/")
            print("saving best model, with val_loss = {}".format(self.best_val_loss))
        print('======================================================================', flush=True)
        return
    
    # helper functions
    def set_train_loss(self):
        for task in self.tasks:
            self.loss_list[task] = []
        self.loss_list['total'] = []
        return
    
    def load_model(self, savePath, reload):
        model_name = True
        for task in self.tasks:
            if task not in reload:
                model_name = False
                print("Breaking...")
                break
        if model_name:
            state = torch.load(savePath + reload)
            self.startIter = state['iter'] + 1
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
        else:
            print('Cannot load from models trained from different tasks.')
            exit()
        return
    
    def save_model(self, it, savePath):
        state = {'iter': it,
                'state_dict': self.model.state_dict(),
                # 'layout': self.model.layout,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
        torch.save(state, savePath + '_'.join(self.tasks) + '.model')
        # if hasattr(self.model, 'branch') and self.model.branch is not None:
        #     torch.save(state, savePath + '_'.join(self.tasks) + '_b' + str(self.model.branch) + '.model')
        # elif hasattr(self.model, 'layout') and self.model.layout is not None:
        #     torch.save(state, savePath + '_'.join(self.tasks) + '.model')
        return
    
    def print_train_loss(self, it):
        # Function: Print loss for each task
        for task in self.tasks:
            if self.loss_list[task]:
                avg_loss = np.mean(self.loss_list[task])
            else:
                continue
            print('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), np.mean(self.loss_list['total'])), flush=True)
        print('======================================================================', flush=True)
        return
    def get_loss(self, it):
        # for task in self.tasks:
        #     if self.loss_list[task]:
        #         avg_loss = np.mean(self.loss_list[task])
        #     else:
        #         continue
        #     print('[Iter {} Task {}] Train Loss: {:.4f}'.format((it+1), task[:4], avg_loss), flush=True)
        print('[Iter {} Total] Train Loss: {:.4f}'.format((it+1), np.mean(self.loss_list['total'])), flush=True)
        # print('======================================================================', flush=True)
        return np.mean(self.loss_list['total'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device = {}".format(device))
# model = model.cuda()
model = model.to(device)

best_path = "/work/sbajaj_umass_edu/BestModels/" +  "_".join(three_task) + '.model'
trainer = Trainer(model, three_task, trainDataloader, valDataloader, criterionDict, metricDict, print_iters=10, val_iters=10, quantization=False, save_iters=200)

savePath = "/work/sbajaj_umass_edu/SavedModels/"
reload = "_".join(three_task) + '.model'
trainer.train(600, loss_lambda, savePath, None)


# device = torch.device('cpu')
# model_fp32 = model.to(device)

# model_fp32.eval()

# model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

# # print(model_fp32)
# model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False)


# with open("validation_results.txt", 'a+') as f:
#     f.write('\n')
#     f.write("Tasks = {}".format("_".join(three_task)))

# startIter = 1
# print('prequantization')
# validate(model_fp32_prepared, valDataloader, three_task, criterionDict, metricDict, startIter)
# model_quantized = torch.quantization.convert(model_fp32_prepared) 
# print('postquantization')
# validate(model_quantized, valDataloader, three_task, criterionDict, metricDict, startIter)