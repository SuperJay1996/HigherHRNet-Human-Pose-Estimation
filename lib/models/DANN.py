from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU
import models

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply

class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class DANN(nn.Module):
    def __init__(self, cfg):
        super(DANN, self).__init__()
        in_nc = cfg.DOMAIN_MODEL.INPUT
        grad_weight = cfg.DOMAIN_MODEL.GRAD_WEIGHT
        num_classes = cfg.DOMAIN_MODEL.NUM_CLASSES
        mid_nc = cfg.DOMAIN_MODEL.MIDDLE

        self.grl = GradientScalarLayer(grad_weight)

        self.dc = nn.Sequential(
            nn.Conv2d(in_nc, mid_nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_nc),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_nc, mid_nc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_nc),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(mid_nc, num_classes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.grl(x)
        x = self.dc(x)
        return x

    def init_weight(self):
        logger.info("=> init DANN weights from normal distribution!")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
    
def get_DA_net(cfg, is_train, **kwargs):
    DA = DANN(cfg)

    if is_train:
        DA.init_weight()

    return DA