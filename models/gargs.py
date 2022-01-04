# -*- coding: utf-8 -*-
from collections import OrderedDict
def init():
    global _args
    _args = OrderedDict()

def set_value(key, value):
    _args[key] = value

def get_value(key):
    try:
        return _args[key]
    except:
        print('error!')
        exit(-1)
