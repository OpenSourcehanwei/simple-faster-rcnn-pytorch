# -*- coding: utf-8 -*-
# Author : hanwei
# Date:  19-4-19 
# Email:  hanwei2008123@hotmail.com
# Filename: predictor.py 
# Copyright (c) 2019 Chengdu Lanjing Data&Information Co., Ltd

'''
预测代码
'''

import torch

from model import FasterRCNNVGG16
from utils.vis_tool import visdom_bbox
from utils import array_tool as at, model_utils


class Predictor:

    def __init__(self, p_model, device):
        self.faster_rcnn = FasterRCNNVGG16()
        self.device = device
        self._load_model(p_model)

    def _load_model(self, p_model):
        state_dict = torch.load(p_model)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)

        self.faster_rcnn.to(self.device)

    def predict(self, img):
        _bboxes, _labels, _scores = self.faster_rcnn.predict([img], visualize=True)
        pred_img = visdom_bbox(img,
                               at.tonumpy(_bboxes[0]),
                               at.tonumpy(_labels[0]).reshape(-1),
                               at.tonumpy(_scores[0]))

        pred_img.savefig('dataset/result/pred_img.png')

if __name__ == '__main__':
    import sys
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device, device_ids = model_utils.get_gpu_devices(0)
    frcnn_predictor = Predictor(sys.argv[1], device)