# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class YOLOF(SingleStageDetector):
    r"""Implementation of `You Only Look One-level Feature
    <https://arxiv.org/abs/2103.09460>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(YOLOF, self).__init__(backbone, neck, bbox_head, train_cfg,
                                    test_cfg, pretrained)

    def forward_dummy(self, img):
        normalized_cls_score, bbox_reg, cls_score, objectness, dummy_bbox_reg = super().forward_dummy(img)
        import torch.nn.functional as F
        normalized_cls_score = F.sigmoid(normalized_cls_score[0])
        # return cls_score, objectness, bbox_reg
        return normalized_cls_score, bbox_reg
