import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from torchvision.ops.misc import Conv2dNormActivation,MLP
from torchvision.utils import _log_api_usage_once
import escnn.nn as enn
from escnn import gspaces
from ..backbones.e2wrn import Wide_ResNet , FIELD_TYPE , WideBasic
class GeM(nn.Module):
    """Implementation of GeM as in https://github.com/filipradenovic/cnnimageretrieval-pytorch
    """
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)

class myAgg(nn.Module):
    """Implementation of ConvAP as of https://arxiv.org/pdf/2210.10239.pdf

    Args:
        in_channels (int): number of channels in the input of ConvAP
        out_channels (int, optional): number of channels that ConvAP outputs. Defaults to 512.
        s1 (int, optional): spatial height of the adaptive average pooling. Defaults to 2.
        s2 (int, optional): spatial width of the adaptive average pooling. Defaults to 2.
    """
    def __init__(self, in_channels, out_channels=512):
        super(myAgg, self).__init__()
        
        # # the model is equivariant under rotations by 45 degrees, modelled by C8
        # self.r2_act = enn.gspaces.rot2dOnR2(N=8)
        # in_type = enn.nn.FieldType(self.r2_act, in_channels*[self.r2_act.trivial_repr])
        # self.in_type = in_type


        # # convolution 1
        # # first specify the output type of the convolutional layer
        # # we choose 24 feature fields, each transforming under the regular representation of C8
        # out_type = enn.nn.FieldType(self.r2_act, out_channels*[self.r2_act.regular_repr])
        # # self.block1 = enn.nn.SequentialModule(
        # #     # enn.nn.MaskModule(in_type, 29, margin=1),
        # #     enn.nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False),
        # #     enn.nn.InnerBatchNorm(out_type),
        # #     enn.nn.ReLU(out_type, inplace=True)
        # # )
        # self.block1 =  enn.nn.SequentialModule(
        #     enn.nn.R2Conv(in_type, out_type, kernel_size=3,padding=1, bias=False), # 5 , 2
        #     # enn.nn.PointwiseDropout(out_type, p=0.3),
        #     # enn.nn.InnerBatchNorm(out_type),
        #     # enn.nn.ReLU(out_type),
        #     # nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size=3),
        #     # nn.InnerBatchNorm(feat_type_hid),
        #     # nn.ReLU(feat_type_hid),
        #     # nn.R2Conv(feat_type_hid, out_type, kernel_size=3),
        # )
        # # self.conv1 = enn.nn.R2Conv(in_type, out_type, kernel_size=3, padding=1, bias=False)
        # # self.AAP=enn.nn.PointwiseAdaptiveAvgPool(out_type, 2)
        # self.pool=enn.nn.GroupPooling(out_type)
        # self.AAP1 = nn.AdaptiveAvgPool2d((15, 15))
        # # self.channel_pool = nn.Conv2d(in_channels=in_channels, out_channels=512*4, kernel_size=3, bias=True)
        # # self.bn=nn.BatchNorm2d(512*4)
        # # self.conv0=nn.Conv2d(in_channels,128,stride=1,kernel_size=(1))
        # # self.linear = nn.Linear((out_channels), out_channels)
        # # self.bn=nn.BatchNorm2d(512)

        # # self.relu1=nn.ReLU()
        # self.linear1 = nn.Linear(out_channels, out_channels)
        # # self.bn1=nn.BatchNorm1d(out_channels)

        # # self.linear2 = nn.Linear(512*4, out_channels)

        # self.gem = GeM()
        # # self.AAP = nn.AdaptiveAvgPool2d((s1, s2))
        # # self.bn=nn.BatchNorm2d(in_channels)
        self.group=gspaces.rot2dOnR2(N=4)
        self.field_type = enn.FieldType(self.group, in_channels*[self.group.regular_repr])
        # self.field_type_trivial = enn.FieldType(self.group, in_channels*[self.group.trivial_repr])
        # self.norm = enn.NormPool(self.field_type)
        # self.norm=enn.NormNonLinearity(self.field_type)
        # self.norm = nn.BatchNorm1d(in_channels)
        self.pool=enn.PointwiseAdaptiveAvgPool2D(self.field_type,1)
        # self.pool = enn.PointwiseAvgPool2D(self.field_type,8).cuda()
        self.linear = nn.Linear(in_channels, out_channels)
        self.gpool = enn.GroupPooling(self.field_type)



    def forward(self, x):
        # x=self.AAP1(x)
        # x =self.in_type(x)
        # x=self.block1(x)
        # # x=self.AAP(x)
        # x=self.pool(x)

        # x = x.tensor
        # x = self.gem(x)
        # x = x.flatten(1)
        # x= self.linear1(x)
        # # x = self.linear(x.flatten(1))
        # # bs,_,_,_=x.shape
        # # x = F.normalize(x, p=2, dim=1)
        # # x=self.channel_pool(x)
        # # x= self.vit(x)
        # x = F.normalize(x.flatten(1), p=2, dim=1)
       
        # x = self.pool(x)
        # x = self.gpool(x).tensor.flatten(1)
        # x = F.normalize(x, p=2, dim=1)
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1)
        return x


# def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 
if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10)
    m = myAgg(2048, 512)
    r = m(x)
    print(r.shape)