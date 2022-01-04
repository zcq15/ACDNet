import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import numpy as np

import sys

sys.path.append('.')
sys.path.append('..')

from .resnet import ResEncoder
from .upproj import DispUpproj,Trans
from ..modules import _make_pad

class ACDNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.max = 10.0

        layers = 50
        iters = 3
        imgsize = (512,1024)
        encoder = 'acdnet'
        squeeze = 2
        trans = None
        calib = 'squeeze'

        if layers <= 34:
            ch_lst = [64, 64, 128, 256, 512]
        else:
            ch_lst = [64, 256, 512, 1024, 2048]

        self.register_buffer('x_mean', torch.FloatTensor(np.array([0.485, 0.456, 0.406])[None, :, None, None]))
        self.register_buffer('x_std', torch.FloatTensor(np.array([0.229, 0.224, 0.225])[None, :, None, None]))

        self.equi = ResEncoder(layers,in_channels=in_channels,output_size=imgsize,encoder=encoder,pretrained=True,log=True)
        layer0 = list(self.equi.layer0.children())
        delattr(self.equi, 'layer0')
        self.equi.conv = nn.Sequential(*layer0[:-2])
        self.equi.pool = nn.Sequential(*layer0[-2:])

        self.ch_up = [(ch_lst[-1]//2**i)//squeeze for i in range(5)]
        self.ch_skip = [_ for _ in ch_lst[3::-1]+[0]]
        if not trans is None: self.ch_skip = [_//squeeze for _ in self.ch_skip]

        self.ch_depth = [0]*(5-iters) + [1]*iters   # default [0,0,1,1,1]
        self.pred = [False]*(5-iters-1) + [True]*(iters+1) # default [False,True,True,True,True]

        self.trans = nn.ModuleList(
            [Trans(in_channels=c,block=trans,out_channels=c//squeeze) for c in ch_lst[3::-1]+[0]]
        )
        self.calib = Trans(in_channels=ch_lst[-1],block=calib,out_channels=ch_lst[-1]//squeeze)

        self.depth = nn.ModuleList(
            [DispUpproj(1,self.ch_up[d],self.ch_skip[d],self.ch_depth[d],self.pred[d]) for d in range(5)]
        )
        
        self.relu = nn.ReLU(inplace=False)

        self.set_parameters(pars='all')
        torch.cuda.empty_cache()

    def extract_feat(self,encoder,x):
        feats = [None]
        feat = encoder.conv(x) # H/2
        feats = [feat]+feats
        feat = encoder.pool(feat) # H/4
        for e in range(1,5):
            feat = getattr(encoder, 'layer%d'%e)(feat)
            if e < 4:
                feats = [feat]+feats
        feat = self.calib(feat)
        return feat,feats

    def trans_feat(self,trans,feats):
        for i in range(len(feats)):
            feats[i] = trans[i](feats[i])
        return feats

    def decode_feat(self,mode,decoder,feat,feats,_output,mid_key):
        if mode=='train': _output[mid_key] = []
        mid = None
        for d in range(5):
            if not mid is None:
                b,c,h,w = mid.shape
                mid_up = F.interpolate(mid,size=(h*2,w*2),mode='bilinear',align_corners=False)
            else:
                mid_up = None
            feat,mid = decoder[d](feat,feats[d],mid_up)
            if not mid is None and mode =='train':
                _output[mid_key] = [torch.sigmoid(mid)*self.max]+_output[mid_key]
        return mid

    def forward(self, _input, mode='test', **kwards):
        if isinstance(_input,OrderedDict) or isinstance(_input,dict):
            equi = (_input['rgb'] - self.x_mean) / self.x_std
        elif torch.is_tensor(_input):
            equi = _input
        else:
            print('Input must be tensor or dict !')
            exit(-1)

        feat_equi,feats_equi = self.extract_feat(self.equi,equi)
        _output = OrderedDict()

        feats_equi = self.trans_feat(self.trans,feats_equi)

        depth = self.decode_feat(mode,self.depth,feat_equi,feats_equi,_output,'ms_depth')

        # _output['depth'] = torch.sigmoid(depth)*self.max
        _output['depth'] = self.relu(depth)
        
        return _output

    def predict(self, _input, **kwards):
        with torch.no_grad():
            return self.forward(_input, mode='test')

    def set_parameters(self,pars='all'):
        if pars == 'all':
            for par in self.parameters():
                par.requires_grad = True
