from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import models

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class PoseDAResNet(nn.Module):
    def __init__(self, cfg, is_train):
        super(PoseDAResNet, self).__init__()

        self.pose = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
            cfg, is_train=is_train
        )

        self.DA = eval('models.'+cfg.DOMAIN_MODEL.NAME+'.get_DA_net')(
            cfg, is_train=is_train
        )

    def forward(self, x):
        final_outputs, domain_feature = self.pose(x)
        
        domain_features = self.DA(domain_feature)
        # print(domain_feature.shape)

        return final_outputs, domain_features
    
    def init_weights(self, pretrained='', verbose=True):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        parameters_names = set()
        for name, _ in self.named_parameters():
            parameters_names.add(name)

        buffers_names = set()
        for name, _ in self.named_buffers():
            buffers_names.add(name)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    if name in parameters_names or name in buffers_names:
                        if verbose:
                            logger.info(
                                '=> init {} from {}'.format(name, pretrained)
                            )
                        need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseDAResNet(cfg, is_train, **kwargs)

    if is_train and cfg.TOTAL_MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.TOTAL_MODEL.PRETRAINED, verbose=cfg.VERBOSE)

    return model