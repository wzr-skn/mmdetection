# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from mmdet.utils import log_img_scale
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors import SingleStageDetector
from mmcv.cnn import ConvModule
from torch.utils.data import DataLoader

@DETECTORS.register_module()
class PPYOLOE(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 input_size=(320, 320),
                 size_multiplier=32,
                 random_size_range=(10, 10),
                 random_size_interval=10,
                 init_cfg=None):
        super(PPYOLOE, self).__init__(backbone, neck, bbox_head, train_cfg,
                                      test_cfg, pretrained, init_cfg)
        log_img_scale(input_size, skip_square=True)
        self.rank, self.world_size = get_dist_info()
        self._default_input_size = input_size
        self._input_size = input_size
        self._random_size_range = random_size_range
        self._random_size_interval = random_size_interval
        self._size_multiplier = size_multiplier
        self._progress_in_iter = 0

    # def forward_train(self,
    #                   img,
    #                   img_metas,
    #                   gt_bboxes,
    #                   gt_labels,
    #                   gt_bboxes_ignore=None):
    #     """
    #     Args:
    #         img (Tensor): Input images of shape (N, C, H, W).
    #             Typically these should be mean centered and std scaled.
    #         img_metas (list[dict]): A List of image info dict where each dict
    #             has: 'img_shape', 'scale_factor', 'flip', and may also contain
    #             'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
    #             For details on the values of these keys see
    #             :class:`mmdet.datasets.pipelines.Collect`.
    #         gt_bboxes (list[Tensor]): Each item are the truth boxes for each
    #             image in [tl_x, tl_y, br_x, br_y] format.
    #         gt_labels (list[Tensor]): Class indices corresponding to each box
    #         gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
    #             boxes can be ignored when computing the loss.
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     # Multi-scale training
    #     img, gt_bboxes = self._preprocess(img, gt_bboxes)
    #
    #     losses = super(PPYOLOE, self).forward_train(img, img_metas, gt_bboxes,
    #                                                 gt_labels, gt_bboxes_ignore)
    #
    #     # random resizing
    #     if (self._progress_in_iter + 1) % self._random_size_interval == 0:
    #         self._input_size = self._random_resize()
    #     self._progress_in_iter += 1
    #
    #     return losses
    #
    # def _preprocess(self, img, gt_bboxes):
    #     self._default_input_size = (img.shape[2], img.shape[3])
    #     scale_y = self._input_size[0] / self._default_input_size[0]
    #     scale_x = self._input_size[1] / self._default_input_size[1]
    #     if scale_x != 1 or scale_y != 1:
    #         img = F.interpolate(
    #             img,
    #             size=self._input_size,
    #             mode='bilinear',
    #             align_corners=False)
    #         for gt_bbox in gt_bboxes:
    #             gt_bbox[..., 0::2] = gt_bbox[..., 0::2] * scale_x
    #             gt_bbox[..., 1::2] = gt_bbox[..., 1::2] * scale_y
    #     return img, gt_bboxes
    #
    # def _random_resize(self):
    #     tensor = torch.LongTensor(2).cuda(1)
    #
    #     if self.rank == 0:
    #         size = random.randint(*self._random_size_range)
    #         aspect_ratio = float(
    #             self._default_input_size[1]) / self._default_input_size[0]
    #         size = (self._size_multiplier * size,
    #                 self._size_multiplier * int(aspect_ratio * size))
    #         tensor[0] = size[0]
    #         tensor[1] = size[1]
    #
    #     if self.world_size > 1:
    #         dist.barrier()
    #         dist.broadcast(tensor, 0)
    #
    #     input_size = (tensor[0].item(), tensor[1].item())
    #     return input_size


    def forward_dummy(self, img):
        cls_score, bbox_reg = super().forward_dummy(img)
        # import torch.nn.functional as F
        # for i in range(len(cls_score)):
        #     cls_score[i].sigmoid_()
        #     objectness[i].sigmoid_()
        return cls_score, bbox_reg