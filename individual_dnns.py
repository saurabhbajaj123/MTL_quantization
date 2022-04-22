import pwd
import numpy as np
import os
import copy
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from tqdm import tqdm

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

new_model = True
perform_quantization = False

tasks = ('segment_semantic','normal','depth_zbuffer')
task_cls_num = {'segment_semantic': 40, 'normal':3, 'depth_zbuffer': 1}

criterionDict = {}
metricDict = {}
clsNum = {}
# three_task = ['segment_semantic','normal','depth_zbuffer']
three_task = ['depth_zbuffer']
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


pretrained = models.resnet18(pretrained=True)
pretrained_features = torch.nn.Sequential(*list(pretrained.children())[:-2])


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

model = MTLModel(512, clsNum, pretrained_features)    

loss_lambda = {'segment_semantic': 1, 'normal':1, 'depth_zbuffer': 1}
checkpoint = '/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/trained'

model = model.cuda()


tb = SummaryWriter()

class Trainer():
    def __init__(self, model, tasks, train_dataloader, val_dataloader, criterion_dict, metric_dict, 
                 lr=0.001, decay_lr_freq=4000, decay_lr_rate=0.5,
                 print_iters=50, val_iters=200, save_iters=200, optimizer=None, quantization=False):
        print("initializing the Trainer")
        super(Trainer, self).__init__()
        self.model = model
        self.startIter = 0
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
                tb.add_scalar('Loss', self.get_loss(i), i)
                self.set_train_loss()
            if (i+1) % self.val_iters == 0:
                self.validate(i)
            if (i+1) % self.save_iters == 0:
                if savePath is not None:
                    self.save_model(i, savePath)
                    # torch.save(model, "/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/resnet18_DGMS_MTL_full_trained_K3_1e--2_test2_iters_{}.pt".format(i))
            # print(self.get_loss(i))
            # tb.add_scalar('Loss', self.get_loss(i), i)


            ############## Quantization #####################
            if self.quantization: 
                tb.add_histogram('conv1.weight', model.network.backbone[0].weight, i)
                tb.add_histogram('4.0.conv1.weight', model.network.backbone[4][0].conv1.weight, i)
                tb.add_histogram('4.0.conv2.weight', model.network.backbone[4][0].conv2.weight, i)
                tb.add_histogram('4.1.conv1.weight', model.network.backbone[4][1].conv1.weight, i)
                tb.add_histogram('4.1.conv2.weight', model.network.backbone[4][1].conv2.weight, i)
                # tb.add_histogram('4.2.conv1.weight', model.network.backbone[4][2].conv1.weight, i)
                # tb.add_histogram('4.2.conv2.weight', model.network.backbone[4][2].conv2.weight, i)
                tb.add_histogram('5.0.conv1.weight', model.network.backbone[5][0].conv1.weight, i)
                tb.add_histogram('5.0.conv2.weight', model.network.backbone[5][0].conv2.weight, i)
                tb.add_histogram('5.1.conv1.weight', model.network.backbone[5][1].conv1.weight, i)
                tb.add_histogram('5.1.conv2.weight', model.network.backbone[5][1].conv2.weight, i)
                # tb.add_histogram('5.2.conv1.weight', model.network.backbone[5][2].conv1.weight, i)
                # tb.add_histogram('5.2.conv2.weight', model.network.backbone[5][2].conv2.weight, i)
                # tb.add_histogram('5.3.conv1.weight', model.network.backbone[5][3].conv1.weight, i)
                # tb.add_histogram('5.3.conv2.weight', model.network.backbone[5][3].conv2.weight, i)
                tb.add_histogram('6.0.conv1.weight', model.network.backbone[6][0].conv1.weight, i)
                tb.add_histogram('6.0.conv2.weight', model.network.backbone[6][0].conv2.weight, i)
                tb.add_histogram('6.1.conv1.weight', model.network.backbone[6][1].conv1.weight, i)
                tb.add_histogram('6.1.conv2.weight', model.network.backbone[6][1].conv2.weight, i)
                # tb.add_histogram('6.2.conv1.weight', model.network.backbone[6][2].conv1.weight, i)
                # tb.add_histogram('6.2.conv2.weight', model.network.backbone[6][2].conv2.weight, i)
                # tb.add_histogram('6.3.conv1.weight', model.network.backbone[6][3].conv1.weight, i)
                # tb.add_histogram('6.3.conv2.weight', model.network.backbone[6][3].conv2.weight, i)
                # tb.add_histogram('6.4.conv1.weight', model.network.backbone[6][4].conv1.weight, i)
                # tb.add_histogram('6.4.conv2.weight', model.network.backbone[6][4].conv2.weight, i)
                # tb.add_histogram('6.5.conv1.weight', model.network.backbone[6][5].conv1.weight, i)
                # tb.add_histogram('6.5.conv2.weight', model.network.backbone[6][5].conv2.weight, i)
                tb.add_histogram('7.0.conv1.weight', model.network.backbone[7][0].conv1.weight, i)
                tb.add_histogram('7.0.conv2.weight', model.network.backbone[7][0].conv2.weight, i)
                tb.add_histogram('7.1.conv1.weight', model.network.backbone[7][1].conv1.weight, i)
                tb.add_histogram('7.1.conv2.weight', model.network.backbone[7][1].conv2.weight, i)
            else:

                ####### Normal Execution ######
                tb.add_histogram('conv1.weight', model.backbone[0].weight, i)
                tb.add_histogram('4.0.conv1.weight', model.backbone[4][0].conv1.weight, i)
                tb.add_histogram('4.0.conv2.weight', model.backbone[4][0].conv2.weight, i)
                tb.add_histogram('4.1.conv1.weight', model.backbone[4][1].conv1.weight, i)
                tb.add_histogram('4.1.conv2.weight', model.backbone[4][1].conv2.weight, i)
                # tb.add_histogram('4.2.conv1.weight', model.backbone[4][2].conv1.weight, i)
                # tb.add_histogram('4.2.conv2.weight', model.backbone[4][2].conv2.weight, i)
                tb.add_histogram('5.0.conv1.weight', model.backbone[5][0].conv1.weight, i)
                tb.add_histogram('5.0.conv2.weight', model.backbone[5][0].conv2.weight, i)
                tb.add_histogram('5.1.conv1.weight', model.backbone[5][1].conv1.weight, i)
                tb.add_histogram('5.1.conv2.weight', model.backbone[5][1].conv2.weight, i)
                # tb.add_histogram('5.2.conv1.weight', model.backbone[5][2].conv1.weight, i)
                # tb.add_histogram('5.2.conv2.weight', model.backbone[5][2].conv2.weight, i)
                # tb.add_histogram('5.3.conv1.weight', model.backbone[5][3].conv1.weight, i)
                # tb.add_histogram('5.3.conv2.weight', model.backbone[5][3].conv2.weight, i)
                tb.add_histogram('6.0.conv1.weight', model.backbone[6][0].conv1.weight, i)
                tb.add_histogram('6.0.conv2.weight', model.backbone[6][0].conv2.weight, i)
                tb.add_histogram('6.1.conv1.weight', model.backbone[6][1].conv1.weight, i)
                tb.add_histogram('6.1.conv2.weight', model.backbone[6][1].conv2.weight, i)
                # tb.add_histogram('6.2.conv1.weight', model.backbone[6][2].conv1.weight, i)
                # tb.add_histogram('6.2.conv2.weight', model.backbone[6][2].conv2.weight, i)
                # tb.add_histogram('6.3.conv1.weight', model.backbone[6][3].conv1.weight, i)
                # tb.add_histogram('6.3.conv2.weight', model.backbone[6][3].conv2.weight, i)
                # tb.add_histogram('6.4.conv1.weight', model.backbone[6][4].conv1.weight, i)
                # tb.add_histogram('6.4.conv2.weight', model.backbone[6][4].conv2.weight, i)
                # tb.add_histogram('6.5.conv1.weight', model.backbone[6][5].conv1.weight, i)
                # tb.add_histogram('6.5.conv2.weight', model.backbone[6][5].conv2.weight, i)
                tb.add_histogram('7.0.conv1.weight', model.backbone[7][0].conv1.weight, i)
                tb.add_histogram('7.0.conv2.weight', model.backbone[7][0].conv2.weight, i)
                tb.add_histogram('7.1.conv1.weight', model.backbone[7][1].conv1.weight, i)
                tb.add_histogram('7.1.conv2.weight', model.backbone[7][1].conv2.weight, i)


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
        # print("size(x) = ", x.size())
        self.optimizer.zero_grad()
        output = self.model(x)
        # print("output = {}".format(output.size()))
        loss = 0
        for task in self.tasks:
            # print("task = {}".format(task))
            y = data[task].cuda()
            if task + '_mask' in data:
                tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
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
            output = self.model(x)
            
            for task in self.tasks:
                y = data[task].cuda()
                if task + '_mask' in data:
                    tloss = self.criterion_dict[task](output[task], y, data[task + '_mask'].cuda())
                    self.metric_dict[task](output[task], y, data[task + '_mask'].cuda())
                else:
                    tloss = self.criterion_dict[task](output[task], y)
                    self.metric_dict[task](output[task], y)
                loss_list[task].append(tloss.item())
        with open("validation_results.txt", 'a+') as f:
            f.write('______________________________________________________________________')
            f.write("{}_{}.pt".format("resnet18_DGMS_MTL_full", "K3_1e--2_test2"))

        for task in self.tasks:
            val_results = self.metric_dict[task].val_metrics()
            print('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])), flush=True)
            print(val_results, flush=True)
            with open("validation_results.txt", 'a+') as f:
                f.write('\n')
                f.write('[Iter {} Task {}] Val Loss: {:.4f}'.format((it+1), task[:4], np.mean(loss_list[task])))
                f.write('\n')
                f.write(str(val_results))
                f.write('\n')
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
        torch.save(state, savePath + '_' + '_'.join(self.tasks) + '.model')
        
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


trainer = Trainer(model, three_task, trainDataloader, valDataloader, criterionDict, metricDict, print_iters=5, val_iters=199, save_iters=199, quantization=perform_quantization)
trainer.train(400, loss_lambda, checkpoint)