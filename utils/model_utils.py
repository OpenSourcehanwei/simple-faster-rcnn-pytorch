# -*- coding: utf-8 -*-
# Author : hanwei
# Date:  18-8-22 
# Email:  hanwei2008123@hotmail.com
# Filename: model_utils.py 
# Copyright (c) 2018 Chengdu Lanjing Data&Information Co., Ltd

import math
import torch
from imblearn import over_sampling, under_sampling
import numpy as np
from numpy import random


def get_gpu_devices(num_gpus):
    count_gpu = torch.cuda.device_count()
    if num_gpus > 0 and count_gpu > 0:
        if num_gpus > count_gpu:
            raise ValueError("系统可用 GPU 个数为：{}，没有更多的 GPU 满足 num_gpus 需要：{}".format(
                count_gpu, num_gpus))
        device = torch.device('cuda')
        device_ids = list(range(num_gpus))
    else:
        device = torch.device('cpu')
        device_ids = None

    return device, device_ids
