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
import random
import argparse
from pathlib import Path
from copy import deepcopy
import torchvision.models as models
import torchvision
from torch.utils.tensorboard import SummaryWriter
from DGMSParent.utils.sparsity import SparsityMeasure
import DGMSParent.config as cfg
from DGMSParent.modeling.DGMS import DGMSConv
from DGMSParent.utils.PyTransformer.transformers.torchTransformer import TorchTransformer


# from tensorboard import program


from TreeMTL.data.nyuv2_dataloader_adashare import NYU_v2
from TreeMTL.data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from TreeMTL.data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics
from TreeMTL.main.trainer import Trainer
from TreeMTL.main.head import ASPPHeadNode, Classification_Module
# from DGMS.utils.sparsity import SparsityMeasure
# from DGMS.modeling import DGMSNet


new_model = True
perform_quantization = True
model_name = "resnet18_MTL_DGMS_full"
suffix = "K8"

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
if perform_quantization:
    print("performing quantization")
    model = DGMSNet(model)
    _transformer = TorchTransformer()
    _transformer.register(nn.Conv2d, DGMSConv)
    model = _transformer.trans_layers(model)
# print(model)

if new_model:
    print("creating new model")
    def get_optimizer(model):
        # train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr}]
        # optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
        #                             weight_decay=args.weight_decay, nesterov=args.nesterov)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.get_1x_lr_params()), lr=0.001, betas=(0.5, 0.999), weight_decay=0.0001)
        return optimizer
    if perform_quantization:
        print("performing quantization")
        optimizer = get_optimizer(model)
        # print(optimizer)
    print("model creation complete")
    model = model.cuda()
    print("saving the new model")
    torch.save(model, "/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/{}_not_trained_{}.pt".format(model_name, suffix))
    print("model saving complete")
else:
    print("loading model")
    # model = torch.load("/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/{}_iters_{}.pt".format(model_name, iterations))
    model = torch.load("/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/{}_{}.pt".format(model_name, suffix))

    print("finished loading the model")
with open("model.txt", 'w+') as f:
    f.write("{}_not_trained_{}.pt".format(model_name, suffix))
    f.write(str(model))


loss_lambda = {'segment_semantic': 1, 'normal':1, 'depth_zbuffer': 1}
checkpoint = 'Checkpoints/NYUv2/test/'



# # torch.save(model, "/content/drive/MyDrive/Courses/Sem_2_Spring_22/Independent_study/Checkpoints/NYUv2/test/quantized_DGMS_MTL_not_trained.pt")
# torch.save(model, "/content/drive/MyDrive/Courses/Sem_2_Spring_22/Independent_study/Checkpoints/NYUv2/test/resnet18_basic_MTL_trained_199iters.pt")
# model = torch.load("/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/resnet18_quantized_DGMS_MTL_not_trained.pt")
# # model = torch.load("/content/drive/My Drive/Courses/Sem_2_Spring_22/Independent_study/Checkpoints/NYUv2/test/quantizedsegment_semantic_normal_depth_zbuffer.pth")
# # model = torch.load("/content/drive/MyDrive/Courses/Sem_2_Spring_22/Independent_study/Checkpoints/NYUv2/test/resnet18_quantized_DGMS_MTL_trained_199iters.pt")


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
        print(self.model.parameters())
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
                    # self.save_model(i, savePath)
                    torch.save(model, "/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test/{}_{}_iters_{}.pt".format(model_name,suffix, i))
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
        print("starting train step")
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
                'layout': self.model.layout,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict()}
        if hasattr(self.model, 'branch') and self.model.branch is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '_b' + str(self.model.branch) + '.model')
        elif hasattr(self.model, 'layout') and self.model.layout is not None:
            torch.save(state, savePath + '_'.join(self.tasks) + '.model')
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

if perform_quantization:
    print("checking sparsity")
    sparsity = SparsityMeasure(None)
    total_sparse_ratio, model_params, compression_rate = sparsity.check_sparsity_per_layer(model)
    with open("compression_rate.txt", 'a+') as f:
        f.write('\n')
        f.write("Sparsity before training")
        f.write("{}_not_trained_{}.pt".format(model_name, suffix))
        f.write(str(compression_rate))
trainer = Trainer(model, three_task, trainDataloader, valDataloader, criterionDict, metricDict, print_iters=1, val_iters=199, quantization=perform_quantization)
trainer.train(200, loss_lambda, checkpoint)

if perform_quantization:
    print("checking sparsity")
    sparsity = SparsityMeasure(None)
    total_sparse_ratio, model_params, compression_rate = sparsity.check_sparsity_per_layer(model)
    with open("compression_rate.txt", 'a+') as f:
        f.write('\n')
        f.write("sparisity after training")
        f.write("{}_{}.pt".format(model_name, suffix))
        f.write(str(compression_rate))