# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from mmcv.cnn import ConvModule
from torch.utils.data import  DataLoader

@DETECTORS.register_module()
class YOLOX(SingleStageDetector):
    r"""Implementation of `YOLOX: Exceeding YOLO Series in 2021
    <https://arxiv.org/abs/2107.08430>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(YOLOX, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained, init_cfg)

    # def forward_dummy(self, img):
    #     cls_score, bbox_reg, objectness = super().forward_dummy(img)
    #     object_cls_score = []
    #     import torch.nn.functional as F
    #     for i in range(len(cls_score)):
    #         object_cls_score.append(cls_score[i].sigmoid()*objectness[i].sigmoid())
    #     return object_cls_score, bbox_reg

    def forward_dummy(self, img):
        cls_score, bbox_reg, objectness = super().forward_dummy(img)
        # import torch.nn.functional as F
        # for i in range(len(cls_score)):
        #     cls_score[i].sigmoid_()
        #     objectness[i].sigmoid_()
        return cls_score, bbox_reg, objectness