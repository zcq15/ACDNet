import argparse
import os
import time
import math
from collections import OrderedDict

import sys
sys.path.append('.')

from . import gargs

class Options(object):
    def __init__(self):
        # create ArgumentParser() obj
        # formatter_class For customization to help document input formatter-class
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # call add_argument() to add parser
    def init(self):
        # add parser
        self.parser.add_argument('--gpus', type=str, default='0', help='gpus')

        self.parser.add_argument('--checkpoints', type=str, default='./checkpoints/acdnet-m3d.pt', help='models are saved here')
        self.parser.add_argument('--example', type=str, default='./examples/m3d.png', help='example')

    def _init_global(self):
        global gargs
        gargs = OrderedDict()
        
    def paser(self):
        self.init()
        opt = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

        keys = ['gpus','checkpoints','example']
        gargs.init()
        for key in keys:
            gargs._args[key] = getattr(opt,key)

        return opt
