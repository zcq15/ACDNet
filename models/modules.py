import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('.')

class ZeroPad(nn.Module):
    def __init__(self, padding):
        super(ZeroPad, self).__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]
    
    def forward(self, x):
        x = F.pad(x, (self.w, self.w, self.h, self.h)) 
        return x

class CircPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]
    def forward(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h == 0 and self.w == 0:
            return x
        elif self.h == 0:
            return F.pad(x,pad=(self.w,self.w,0,0),mode='circular')
        else:
            idx = torch.arange(-W//2,W//2,device=x.device)
            up = x[:,:,:self.h,idx].flip(2)
            down = x[:,:,-self.h:,idx].flip(2)
            return F.pad(torch.cat([up,x,down],dim=2),pad=(self.w,self.w,0,0),mode='circular')

class LRPad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        if isinstance(padding,int):
            self.h = padding
            self.w = padding
        else:
            self.h = padding[0]
            self.w = padding[1]
    def forward(self,x):
        _,_,H,W = x.shape
        assert H%2==0 and W%2==0
        if self.h==0 and self.w==0:
            return x
        return F.pad(F.pad(x,pad=(self.w,self.w,0,0),mode='circular'),pad=(0,0,self.h,self.h))

def _make_pad(padding=0,pad='lrpad',**kargs):
    if pad == 'circpad':
        return CircPad(padding)
    elif pad == 'lrpad':
        return LRPad(padding)
    else:
        return ZeroPad(padding)

def _make_norm(norm,layers,**kargs):
    if norm is None or norm == 'idt':
        return nn.Identity()
    elif norm == 'bn':
        return nn.BatchNorm2d(layers)
    else:
        raise NotImplementedError

def _make_act(act,**kargs):
    if act is None or act == 'idt':
        return nn.Identity()
    elif act == 'relu':
        return nn.ReLU(inplace=False)
    else:
        raise NotImplementedError
