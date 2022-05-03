import torch
import torch.nn as nn
import os
# from post_training_quantization import MTLModel
from fusing_layers import *
from TreeMTL.data.nyuv2_dataloader_adashare import NYU_v2
from TreeMTL.data.pixel2pixel_loss import NYUCriterions, TaskonomyCriterions
from TreeMTL.data.pixel2pixel_metrics import NYUMetrics, TaskonomyMetrics
from TreeMTL.main.trainer import Trainer
from TreeMTL.main.head import ASPPHeadNode, Classification_Module
import copy

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

task_list = [['segment_semantic'],
['normal'],
['depth_zbuffer'],
['segment_semantic', 'normal'],
['segment_semantic', 'depth_zbuffer'],
['depth_zbuffer', 'normal'],
['segment_semantic','normal','depth_zbuffer']]



def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size

    
def model_compression(t):    
    model_fp32 = torch.load(savePath, map_location=torch.device('cpu'))
    model_fp32.eval()
    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_prepared = torch.quantization.prepare(model_fp32, inplace = False)
    model_quantized = torch.quantization.convert(model_fp32_prepared)
    # compare the sizes
    f=print_size_of_model(model_fp32,"fp32")
    q=print_size_of_model(model_quantized,"int8")
    print("{0:.6f} times smaller".format(f/q))


for t in task_list:
    savePath = "/home/sbajaj/MyCode/quantization_remote/Checkpoints/NYUv2/test2/resnet18_{}_iters_{}.pt".format("_".join(t), 599)
    print(t)
    model_compression(savePath)