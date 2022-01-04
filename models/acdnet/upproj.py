import torch
import torch.nn as nn
import torch.nn.functional as F

import collections
import sys
sys.path.append('.')

from ..modules import _make_pad,_make_norm,_make_act

class FastUpconv(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels//2
        self.conv1_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3,bias=False)),
            ('bn1', _make_norm('bn',out_channels)),
        ]))
        self.conv2_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3),bias=False)),
            ('bn1', _make_norm('bn',out_channels)),
        ]))
        self.conv3_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=(3, 2),bias=False)),
            ('bn1', _make_norm('bn',out_channels)),
        ]))
        self.conv4_ = nn.Sequential(collections.OrderedDict([
            ('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=2,bias=False)),
            ('bn1', _make_norm('bn',out_channels)),
        ]))
        self.ps = nn.PixelShuffle(2)
        # self.norm = _make_norm(_norm,out_channels)
        self.act = _make_act('relu')
    def forward(self, x):
        # print('Upmodule x size = ', x.size())
        x1 = self.conv1_(nn.functional.pad(x, (1, 1, 1, 1)))
        x2 = self.conv2_(nn.functional.pad(x, (1, 1, 0, 1)))
        x3 = self.conv3_(nn.functional.pad(x, (0, 1, 1, 1)))
        x4 = self.conv4_(nn.functional.pad(x, (0, 1, 0, 1)))
        # print(x1.size(), x2.size(), x3.size(), x4.size())
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.ps(x)
        # x = self.norm(x)
        x = self.act(x)
        return x

class DispUpproj(nn.Module):
    def __init__(self,c_out,c_up,c_skip=0,c_mid=0,pred=False):
        super().__init__()
        self.up = nn.Sequential(
            FastUpconv(c_up,c_up//2)
        )

        self.feat = nn.Sequential(
            _make_pad(1),
            nn.Conv2d(c_up//2+c_skip+c_mid,c_up//2,kernel_size=3,bias=False),
            _make_norm('bn',c_up//2),
            _make_act('relu')
            )

        self.use_residual = bool(c_mid)
        if pred:
            self.depth = nn.Sequential(
                _make_pad(1), nn.Conv2d(c_up//2,c_out,kernel_size=3)
            )
        else:
            self.depth = None

        torch.cuda.empty_cache()
    
    def forward(self,x,skip=None,depth=None):
        feats = [self.up(x)]
        if not skip is None:
            feats.append(skip)
        if not depth is None:
            feats.append(depth)
        feat = self.feat(torch.cat(feats,dim=1))
        if not self.depth is None:
            if self.use_residual:
                depth = self.depth(feat)+depth
            else:
                depth = self.depth(feat)
        else:
            depth = None
        return feat,depth

class Squeeze(nn.Module):
    def __init__(self, in_channels, out_channels=None, strict=True):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        if out_channels == in_channels and not strict:
            self.layer = nn.Sequential()
        else:
            self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            _make_norm('bn',out_channels),
            nn.ReLU())
    
    def forward(self,x):
        return self.layer(x)


class Trans(nn.Module):
    def __init__(self, in_channels=0, block=None,**kwargs):
        super().__init__()
        if in_channels <= 0 or block is None:
            self.trans = None
        else:
            self.trans = Squeeze(in_channels=in_channels,**kwargs)
    def forward(self,x):
        if x is None or self.trans is None:
            return x
        else:
            return self.trans(x)