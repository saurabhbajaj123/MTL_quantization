import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import resnet34

from .resnet import resnet
from .mnasnet import *
from .proxylessnas import *
from .resnet_cifar import *
from .vgg_small_cifar import *

def get_network(args):
    return {
        'resnet18': resnet,
        'resnet50': resnet,
        'mnasnet': mnasnet1_0,
        'proxylessnas': proxyless_nas_mobile,
        'vggsmall': vggsmall,
        'resnet20': resnet20,
        'resnet32': resnet32,
        'resnet34': resnet34,
        'resnet56': resnet56,
    }[args.network](args)
