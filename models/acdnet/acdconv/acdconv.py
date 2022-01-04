import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import collections
import sys
sys.path.append('.')
sys.path.append('..')
from ..modules import _make_pad

OmniKernelL = [[5,11],[3,9],[5,7],[7,7]]
OmniKernelS = [[3,3],[3,5],[3,7],[3,9]]
OmniDilationS = [[1,1],[1,2],[1,4],[2,1]]
SK_Groups = 1 # 32
SK_Rate = 16
SK_Length = 32

def xavier_init(m):
    '''Provides Xavier initialization for the network weights and
    normally distributes batch norm params'''
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('ConvTranspose2d') != -1):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)

def kaiming_init(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class PadConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,dilation=1,bias=True,groups=1):
        super().__init__()
        if isinstance(dilation,int):
            dh = dilation
            dw = dilation
        else:
            dh = dilation[0]
            dw = dilation[1]
        if isinstance(kernel_size,int):
            h = (kernel_size // 2)*dh
            w = (kernel_size // 2)*dw
        else:
            h = (kernel_size[0] // 2)*dh
            w = (kernel_size[1] // 2)*dw
        self.pad = _make_pad([h,w])
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,dilation=dilation,bias=bias,groups=groups)
    def forward(self,x):
        return self.conv(self.pad(x))


class ACDConv(nn.Module):
    def __init__(self,in_channels,out_channels=None,dilation_list=OmniDilationS,stride=1,bias=False,groups=SK_Groups,length=SK_Length):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        mid_vectors = max(out_channels//SK_Rate,length)
        self.convs = nn.ModuleList()
        for dilation in dilation_list:
            self.convs.append(nn.Sequential(
                PadConv2d(in_channels,out_channels,kernel_size=3,stride=stride,dilation=dilation,bias=bias,groups=groups),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))

        self.fc = nn.Sequential(
            nn.Conv2d(out_channels,mid_vectors,kernel_size=1,bias=False),
            nn.BatchNorm2d(mid_vectors),
            nn.ReLU(inplace=True)
            )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.ModuleList()
        for _ in dilation_list:
            self.fcs.append(nn.Conv2d(mid_vectors,out_channels,kernel_size=1))

    def forward(self,x):
        feats = []
        for conv in self.convs:
            feats.append(conv(x).unsqueeze(0))
        feats = torch.cat(feats,dim=0) # [N,B,C,H,W]
        vector = self.avg(feats.mean(dim=0)) # [B,C,1,1]
        vector = self.fc(vector) # [B,D,1,1]
        vectors = []
        for fc in self.fcs:
            vectors.append(fc(vector).unsqueeze(0))
        vectors = torch.cat(vectors,dim=0) # [N,B,C,1,1]
        vectors = torch.softmax(vectors,dim=0) # [N,B,C,1,1]
        feat = torch.sum(feats*vectors,dim=0) # [B,C,H,W]
        return feat