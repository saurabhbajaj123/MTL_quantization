import torch
import torch.nn as nn

from .base_node import ComputeNode
from .layer_containers import EltwiseOp, AbstractPool

import sys

class InputNode(nn.Module):
    def __init__(self, inputDim):
        super(InputNode, self).__init__()
        self.output = None
        self.outputDim = inputDim
        self.depth = 0
    
    def set_data(self, x):
        self.output = x

class Conv2dNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(Conv2dNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = True
        self.layerParam = 'convolution_param'
        self.paramMapping = {'in_channels': 'need_input',
                             'out_channels': 'num_output',
                             'kernel_size':  'kernel_size',
                             'stride':       'stride',
                             'padding':      'pad',
                             'dilation':     'dilation',
                             'groups':       'group',
                             'bias':         'bias_term'
                             }
        
        self.build_layer()
    
    def set_output_channels(self):
        # Function: Set output channels according to different OpType
        #           outputDim: Int
        self.outputDim = self.basicOp.out_channels
        return
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        opCommand = 'nn.Conv2d(' + ','.join(torchParamList) + ')'    
        self.basicOp = eval(opCommand)
        return
    
class BN2dNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(BN2dNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = True
        self.layerParam = 'batch_norm_param'
        self.paramMapping = {'num_features': 'need_input',
                             'eps':          'eps',
                             'momentum':     'momentum',
                             'affine':       'affine',
                             'track_running_stats': 'track_running_stats'
                             }
        
        self.build_layer()
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        opCommand = 'nn.BatchNorm2d(' + ','.join(torchParamList) + ')'    
        self.basicOp = eval(opCommand)
        return
    
class ReLUNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(ReLUNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = False
        self.layerParam = 'relu_param'
        self.paramMapping = {'inplace': 'inplace'}
        self.build_layer()
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
#         torchParamList.append('inplace=True')
        opCommand = 'nn.ReLU(' + ','.join(torchParamList) + ')'    
        self.basicOp = eval(opCommand)
        return
    
class PoolNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(PoolNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = False
        self.layerParam = 'pooling_param'
        self.paramMapping = {'pool_method':  'pool',
                             'global_pooling':'global_pooling',
                             'kernel_size':  'kernel_size',
                             'stride':       'stride',
                             'padding':      'pad',
                             'ceil_mode':    'round_mode'
                            }
        self.build_layer()
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        opCommand = 'AbstractPool(' + ','.join(torchParamList) + ')'
        self.basicOp = eval(opCommand)
        return
    
class EltNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(EltNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = False
        self.layerParam = 'eltwise_param'
        self.paramMapping = {'operation': 'operation'}
        self.build_layer()
        
    def set_input_channels(self):
        # Function: Get input channels from the outputDim of fatherNodeList but firstly check the dimension consistency
        #           inputDim: Int 
        if self.fatherNodeList[0].outputDim != self.fatherNodeList[1].outputDim:
            sys.exit('The given two bottoms cannot do element-wise operations because of different dimensions.')
        else:
            self.inputDim = self.fatherNodeList[0].outputDim
        return
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        opCommand = 'EltwiseOp(' + ','.join(torchParamList) + ')'    
        self.basicOp = eval(opCommand)
        return
    
    def internal_compute(self):
        # Function: Forward function when commonly used with 2 bottoms
        return self.basicOp(self.fatherNodeList[0].output, self.fatherNodeList[1].output)
    
class DropoutNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(DropoutNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = False
        self.layerParam = 'dropout_param'
        self.paramMapping = {'p': 'dropout_ratio'}
        
        self.build_layer()
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        opCommand = 'nn.Dropout(' + ','.join(torchParamList) + ')'    
        self.basicOp = eval(opCommand)
        return
    
class LinearNode(ComputeNode):
    def __init__(self, protoLayer, fatherNodeList):
        super(LinearNode, self).__init__(protoLayer, fatherNodeList)
        self.paramOp = True
        self.layerParam = 'inner_product_param'
        self.paramMapping = {'in_features': 'need_input',
                             'out_features': 'num_output',
                             'bias':         'bias_term'
                             }
        
        self.build_layer()
    
    def set_output_channels(self):
        # Function: Set output channels according to different OpType
        #           outputDim: Int
        self.outputDim = self.basicOp.out_features
        return
    
    def generate_basicOp(self, torchParamList):
        # Function: Generate opCommand according to different OpType
        opCommand = 'nn.Linear(' + ','.join(torchParamList) + ')'    
        self.basicOp = eval(opCommand)
        return
    
    