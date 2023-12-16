# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from ..builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..models.assigners.atss_assigner import ATSSAssigner
from ..models.assigners.task_aligned_assigner import TaskAlignedAssigner
import copy
import torchvision
from mmcv.ops.nms import batched_nms


class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = torch.maximum(x1, x1g)
        ykis1 = torch.maximum(y1, y1g)
        xkis2 = torch.minimum(x2, x2g)
        ykis2 = torch.minimum(y2, y2g)
        w_inter = F.relu(xkis2 - xkis1)
        h_inter = F.relu(ykis2 - ykis1)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        # x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        # x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        # torch的split和paddle有点不同，torch的第二个参数表示的是每一份的大小，paddle的第二个参数表示的是分成几份。
        x1, y1, x2, y2 = torch.split(pbox, split_size_or_sections=1, dim=-1)
        x1g, y1g, x2g, y2g = torch.split(gbox, split_size_or_sections=1, dim=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = torch.minimum(x1, x1g)
        yc1 = torch.minimum(y1, y1g)
        xc2 = torch.maximum(x2, x2g)
        yc2 = torch.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = torch.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = torch.sum(giou * iou_weight)
        else:
            loss = torch.mean(giou * iou_weight)
        return loss * self.loss_weight


def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)


class ConvBNLayer(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 filter_size=3,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None):
        super(ConvBNLayer, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=ch_in,
            out_channels=ch_out,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


def normal_(tensor, mean=0., std=1.):
    """
    Modified tensor inspace using normal_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        mean (float|int): mean value.
        std (float|int): std value.
    Return:
        tensor
    """
    return _no_grad_normal_(tensor, mean, std)


def _no_grad_normal_(tensor, mean=0., std=1.):
    with torch.no_grad():
        tensor.copy_(torch.normal(mean=mean, std=std, size=tensor.shape))
    return tensor


class ESEAttn(nn.Module):
    def __init__(self, feat_channels, act='relu'):
        super(ESEAttn, self).__init__()
        self.fc = nn.Conv2d(feat_channels, feat_channels, 1)
        self.conv = ConvBNLayer(feat_channels, feat_channels, 1, act=act)

        self._init_weights()

    def _init_weights(self):
        normal_(self.fc.weight, std=0.001)

    def forward(self, feat, avg_feat):
        weight = torch.sigmoid(self.fc(avg_feat))
        return self.conv(feat * weight)


def constant_(tensor, value=0.):
    """
    Modified tensor inspace using constant_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        value (float|int): value to fill tensor.
    Return:
        tensor
    """
    return _no_grad_fill_(tensor, value)


def _no_grad_fill_(tensor, value=0.):
    with torch.no_grad():
        tensor.copy_(torch.full_like(tensor, value, dtype=tensor.dtype))
    return tensor


def generate_anchors_for_grid_cell(feats,
                                   fpn_strides,
                                   grid_cell_size=5.0,
                                   grid_cell_offset=0.5):
    r"""
    Like ATSS, generate anchors based on grid size.
    Args:
        feats (List[Tensor]): shape[s, (b, c, h, w)]
        fpn_strides (tuple|list): shape[s], stride for each scale feature
        grid_cell_size (float): anchor size
        grid_cell_offset (float): The range is between 0 and 1.
    Returns:
        anchors (Tensor): shape[l, 4], "xmin, ymin, xmax, ymax" format.
        anchor_points (Tensor): shape[l, 2], "x, y" format.
        num_anchors_list (List[int]): shape[s], contains [s_1, s_2, ...].
        stride_tensor (Tensor): shape[l, 1], contains the stride for each scale.
    """
    assert len(feats) == len(fpn_strides)
    anchors = []
    anchor_points = []
    num_anchors_list = []
    stride_tensor = []
    for feat, stride in zip(feats, fpn_strides):
        _, _, h, w = feat.shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(end=w) + grid_cell_offset) * stride
        shift_y = (torch.arange(end=h) + grid_cell_offset) * stride
        # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
        anchor = torch.stack(
            [
                shift_x - cell_half_size, shift_y - cell_half_size,
                shift_x + cell_half_size, shift_y + cell_half_size
            ],
            -1).to(feat.dtype)
        anchor_point = torch.stack(
            [shift_x, shift_y], -1).to(feat.dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(
            torch.full(
                [num_anchors_list[-1], 1], stride, dtype=feat.dtype))
    anchors = torch.cat(anchors)
    anchors.requires_grad_(False)
    anchor_points = torch.cat(anchor_points)
    anchor_points.requires_grad_(False)
    stride_tensor = torch.cat(stride_tensor)
    stride_tensor.requires_grad_(False)
    return anchors, anchor_points, num_anchors_list, stride_tensor


def get_static_shape(tensor):
    # shape = torch.shape(tensor)
    # shape.requires_grad_(False)
    # return shape
    return tensor.shape


def batch_distance2bbox(points, distance, max_shapes=None):
    """Decode distance prediction to bounding box for batch.
    Args:
        points (Tensor): [B, ..., 2], "xy" format
        distance (Tensor): [B, ..., 4], "ltrb" format
        max_shapes (Tensor): [B, 2], "h,w" format, Shape of the image.
    Returns:
        Tensor: Decoded bboxes, "x1y1x2y2" format.
    """
    lt, rb = torch.split(distance, 2, -1)
    # while tensor add parameters, parameters should be better placed on the second place
    x1y1 = -lt + points
    x2y2 = rb + points
    out_bbox = torch.cat([x1y1, x2y2], -1)
    if max_shapes is not None:
        max_shapes = max_shapes.flip(-1).tile([1, 2])
        delta_dim = out_bbox.ndim - max_shapes.ndim
        for _ in range(delta_dim):
            max_shapes.unsqueeze_(1)
        out_bbox = torch.where(out_bbox < max_shapes, out_bbox, max_shapes)
        out_bbox = torch.where(out_bbox > 0, out_bbox, torch.zeros_like(out_bbox))
    return out_bbox


def matrix_nms(bboxes,
               scores,
               score_threshold,
               post_threshold,
               nms_top_k,
               keep_top_k,
               use_gaussian=False,
               gaussian_sigma=2.):
    inds = (scores > score_threshold)
    cate_scores = scores[inds]
    if len(cate_scores) == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0

    inds = inds.nonzero()
    cate_labels = inds[:, 1]
    bboxes = bboxes[inds[:, 0]]

    # sort and keep top nms_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if nms_top_k > 0 and len(sort_inds) > nms_top_k:
        sort_inds = sort_inds[:nms_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    # Matrix NMS
    kernel = 'gaussian' if use_gaussian else 'linear'
    cate_scores = _matrix_nms(bboxes, cate_labels, cate_scores, kernel=kernel, sigma=gaussian_sigma)

    # filter.
    keep = cate_scores >= post_threshold
    if keep.sum() == 0:
        return torch.zeros((1, 6), device=bboxes.device) - 1.0
    bboxes = bboxes[keep, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    # sort and keep keep_top_k
    sort_inds = torch.argsort(cate_scores, descending=True)
    if len(sort_inds) > keep_top_k:
        sort_inds = sort_inds[:keep_top_k]
    bboxes = bboxes[sort_inds, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = cate_scores.unsqueeze(1)
    cate_labels = cate_labels.unsqueeze(1).float()
    pred = torch.cat([cate_labels, cate_scores, bboxes], 1)

    return pred


def _matrix_nms(bboxes, cate_labels, cate_scores, kernel='gaussian', sigma=2.0):
    """Matrix NMS for multi-class bboxes.
    Args:
        bboxes (Tensor): shape (n, 4)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gaussian'
        sigma (float): std in gaussian method
    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    # 计算一个n×n的IOU矩阵，两组矩形两两之间的IOU
    iou_matrix = jaccard(bboxes, bboxes)   # shape: [n_samples, n_samples]
    iou_matrix = iou_matrix.triu(diagonal=1)   # 只取上三角部分

    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)   # shape: [n_samples, n_samples]
    # 第i行第j列表示的是第i个预测框和第j个预测框的类别id是否相同。我们抑制的是同类的预测框。
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)   # shape: [n_samples, n_samples]

    # IoU compensation
    # 非同类的iou置为0，同类的iou保留。逐列取最大iou
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)   # shape: [n_samples, ]
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)   # shape: [n_samples, n_samples]

    # IoU decay
    # 非同类的iou置为0，同类的iou保留。
    decay_iou = iou_matrix * label_matrix   # shape: [n_samples, n_samples]

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1-decay_iou)/(1-compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # 更新分数
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


def jaccard(box_a, box_b):
    """计算两组矩形两两之间的iou
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
        ious: (tensor) Shape: [A, B]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A, B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A, B]
    union = area_a + area_b - inter
    return inter / union  # [A, B]


# 相交矩形的面积
def intersect(box_a, box_b):
    """计算两组矩形两两之间相交区域的面积
    Args:
        box_a: (tensor) bounding boxes, Shape: [A, 4].
        box_b: (tensor) bounding boxes, Shape: [B, 4].
    Return:
      (tensor) intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


# 参考自mmdet/utils/boxes.py的postprocess()。为了保持和matrix_nms()一样的返回风格，重新写一下。
def my_multiclass_nms(bboxes, scores, score_threshold=0.7, nms_threshold=0.45, nms_top_k=1000, keep_top_k=100, class_agnostic=False):
    '''
    :param bboxes:   shape = [N, A,  4]   "左上角xy + 右下角xy"格式
    :param scores:   shape = [N, A, 80]
    :param score_threshold:
    :param nms_threshold:
    :param nms_top_k:
    :param keep_top_k:
    :param class_agnostic:
    :return:
    '''

    # 每张图片的预测结果
    output = [None for _ in range(len(bboxes))]
    # 每张图片分开遍历
    for i, (xyxy, score) in enumerate(zip(bboxes, scores)):
        '''
        :var xyxy:    shape = [A, 4]   "左上角xy + 右下角xy"格式
        :var score:   shape = [A, 80]
        '''

        # 每个预测框最高得分的分数和对应的类别id
        class_conf, class_pred = torch.max(score, 1, keepdim=True)

        # 分数超过阈值的预测框为True
        conf_mask = (class_conf.squeeze() >= score_threshold).squeeze()
        # 这样排序 (x1, y1, x2, y2, 得分, 类别id)
        detections = torch.cat((xyxy, class_conf, class_pred.float()), 1)
        # 只保留超过阈值的预测框
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        # 使用torchvision自带的nms、batched_nms
        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4],
                nms_threshold,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4],
                detections[:, 5],
                nms_threshold,
            )

        detections = detections[nms_out_index]

        # 保留得分最高的keep_top_k个
        sort_inds = torch.argsort(detections[:, 4], descending=True)
        if keep_top_k > 0 and len(sort_inds) > keep_top_k:
            sort_inds = sort_inds[:keep_top_k]
        detections = detections[sort_inds, :]

        # 为了保持和matrix_nms()一样的返回风格 cls、score、xyxy。
        detections = torch.cat((detections[:, 5:6], detections[:, 4:5], detections[:, :4]), 1)

        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output


@HEADS.register_module()
class PPYOLOEHead(BaseDenseHead, BBoxTestMixin):

    def __init__(self,
                 in_channels=[1024, 512, 256],
                 num_classes=80,
                 act='relu',
                 fpn_strides=(32, 16, 8),
                 grid_cell_scale=5.0,
                 grid_cell_offset=0.5,
                 reg_max=16,
                 static_assigner_epoch=4,
                 use_varifocal_loss=True,
                 static_assigner=dict(
                    topk=9,
                    num_classes=1,
                 ),
                 assigner=dict(
                    topk=13,
                    alpha=1.0,
                    beta=6.0,
                 ),
                 nms='MultiClassNMS',
                 eval_size=None,
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                 },
                 train_cfg=None,
                 test_cfg=None,
                 trt=False,
                 nms_cfg=None,
                 exclude_nms=False):
        super(PPYOLOEHead, self).__init__()
        assert len(in_channels) > 0, "len(in_channels) should > 0"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        self.grid_cell_scale = grid_cell_scale
        self.grid_cell_offset = grid_cell_offset
        self.reg_max = reg_max
        self.iou_loss = GIoULoss()
        self.loss_weight = loss_weight
        self.use_varifocal_loss = use_varifocal_loss
        self.eval_size = eval_size

        self.static_assigner_epoch = static_assigner_epoch
        self.static_assigner = ATSSAssigner(**static_assigner)
        self.assigner = TaskAlignedAssigner(**assigner)
        self.nms = nms
        # if isinstance(self.nms, MultiClassNMS) and trt:
        #     self.nms.trt = trt
        self.exclude_nms = exclude_nms
        self.nms_cfg = nms_cfg
        # stem
        self.stem_cls = nn.ModuleList()
        self.stem_reg = nn.ModuleList()
        act = get_act_fn(
            act, trt=trt) if act is None or isinstance(act,
                                                       (str, dict)) else act
        for in_c in self.in_channels:
            self.stem_cls.append(ESEAttn(in_c, act=act))
            self.stem_reg.append(ESEAttn(in_c, act=act))
        # pred head
        self.pred_cls = nn.ModuleList()
        self.pred_reg = nn.ModuleList()
        for in_c in self.in_channels:
            self.pred_cls.append(
                nn.Conv2d(
                    in_c, self.num_classes, 3, padding=1))
            self.pred_reg.append(
                nn.Conv2d(
                    in_c, 4 * (self.reg_max + 1), 3, padding=1))
        # projection conv
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self._init_weights()
        self.epoch = 0  # which would be update in SetEpochInfoHook!

    def _init_weights(self):
        bias_cls = bias_init_with_prob(0.01)
        for cls_, reg_ in zip(self.pred_cls, self.pred_reg):
            constant_(cls_.weight)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.weight)
            constant_(reg_.bias, 1.0)

        self.proj = torch.linspace(0, self.reg_max, self.reg_max + 1)
        self.proj.requires_grad = False
        self.proj_conv.weight.requires_grad_(False)
        self.proj_conv.weight.copy_(
            self.proj.reshape([1, self.reg_max + 1, 1, 1]))

        if self.eval_size:
            anchor_points, stride_tensor = self._generate_anchors()
            self.register_buffer('anchor_points', anchor_points)
            self.register_buffer('stride_tensor', stride_tensor)

    def _generate_anchors(self, feats=None):
        # just use in eval time
        anchor_points = []
        stride_tensor = []
        for i, stride in enumerate(self.fpn_strides):
            if feats is not None:
                _, _, h, w = feats[i].shape
            else:
                h = int(self.eval_size[0] / stride)
                w = int(self.eval_size[1] / stride)
            shift_x = torch.arange(end=w) + self.grid_cell_offset
            shift_y = torch.arange(end=h) + self.grid_cell_offset
            # shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x)
            anchor_point = torch.stack([shift_x, shift_y], -1).to(torch.float32)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(
                torch.full(
                    [h * w, 1], stride, dtype=torch.float32))
        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    def forward(self, feats):
        anchors, anchor_points, num_anchors_list, stride_tensor = \
            generate_anchors_for_grid_cell(
                feats, self.fpn_strides, self.grid_cell_scale,
                self.grid_cell_offset)

        cls_score_list, reg_distri_list = [], []
        for i, feat in enumerate(feats):
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_logit = self.pred_cls[i](self.stem_cls[i](feat, avg_feat) +
                                         feat)
            reg_distri = self.pred_reg[i](self.stem_reg[i](feat, avg_feat))
            # cls and reg
            cls_score = torch.sigmoid(cls_logit)
            cls_score_list.append(cls_score.flatten(2).permute((0, 2, 1)))
            reg_distri_list.append(reg_distri.flatten(2).permute((0, 2, 1)))
        # cls_score_list = torch.cat(cls_score_list, 1)
        # reg_distri_list = torch.cat(reg_distri_list, 1)

        return (cls_score_list, reg_distri_list, anchors, anchor_points,
            num_anchors_list, stride_tensor)

    @staticmethod
    def _focal_loss(score, label, alpha=0.25, gamma=2.0):
        weight = (score - label).pow(gamma)
        if alpha > 0:
            alpha_t = alpha * label + (1 - alpha) * (1 - label)
            weight *= alpha_t

        # loss = F.binary_cross_entropy(
        #     score, label, weight=weight, reduction='sum')

        score = score.to(torch.float32)
        eps = 1e-9
        loss = label * (0 - torch.log(score + eps)) + \
               (1.0 - label) * (0 - torch.log(1.0 - score + eps))
        loss *= weight
        loss = loss.sum()
        return loss

    @staticmethod
    def _varifocal_loss(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label

        # loss = F.binary_cross_entropy(
        #     pred_score, gt_score, weight=weight, reduction='sum')

        # pytorch的F.binary_cross_entropy()的weight不能向前传播梯度，但是
        # paddle的F.binary_cross_entropy()的weight可以向前传播梯度（给pred_score），
        # 所以这里手动实现F.binary_cross_entropy()
        # 使用混合精度训练时，pred_score类型是torch.float16，需要转成torch.float32避免log(0)=nan
        pred_score = pred_score.to(torch.float32)
        eps = 1e-9
        loss = gt_score * (0 - torch.log(pred_score + eps)) + \
               (1.0 - gt_score) * (0 - torch.log(1.0 - pred_score + eps))
        loss *= weight
        loss = loss.sum()
        return loss


    @force_fp32(apply_to=('pred_scores', 'pred_distri'))
    def get_bboxes(self,
                   pred_scores,
                   pred_distri,
                   anchors,
                   anchor_points,
                   num_anchors_list,
                   stride_tensor,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        pred_scores_list, pred_distri_list = [], []
        for i in range(len(pred_scores)):
            b, l, _ = pred_scores[i].shape
            pred_scores_list.append(pred_scores[i].permute((0, 2, 1)))
            reg_dist = pred_distri[i].reshape([-1, 4, self.reg_max + 1, l])
            reg_dist = reg_dist.permute((0, 2, 1, 3))
            reg_dist = self.proj_conv(F.softmax(reg_dist, dim=1))
            pred_distri_list.append(reg_dist.reshape([b, 4, l]))

        pred_scores_list = torch.cat(pred_scores_list, -1)
        pred_distri_list = torch.cat(pred_distri_list, -1)

        device = anchor_points.device
        pred_scores_list = pred_scores_list.to(device)
        pred_distri_list = pred_distri_list.to(device)

        pred_distri_list = batch_distance2bbox(anchor_points, pred_distri_list.permute((0, 2, 1)))
        pred_distri_list *= stride_tensor
        # scale bbox to origin
        scale_factor = torch.tensor(
            [img_meta['scale_factor'] for img_meta in img_metas]).reshape([-1, 1, 4])
        pred_distri_list /= scale_factor  # [N, A, 4]     pred_scores.shape = [N, 80, A]
        if self.exclude_nms:
            # `exclude_nms=True` just use in benchmark
            return pred_distri_list.sum(), pred_scores_list.sum()
        else:
            # nms
            preds = []
            nms_cfg = copy.deepcopy(self.nms_cfg)
            nms_type = nms_cfg.pop('nms_type')
            batch_size = pred_distri_list.shape[0]
            yolo_scores = pred_scores_list.permute((0, 2, 1))  # [N, A, 80]
            if nms_type == 'matrix_nms':
                for i in range(batch_size):
                    pred = matrix_nms(pred_distri_list[i, :, :], yolo_scores[i, :, :], **nms_cfg)
                    preds.append(pred)
            elif nms_type == 'multiclass_nms':
                preds = my_multiclass_nms(pred_distri_list, yolo_scores, **nms_cfg)
            elif nms_type == 'bboxes_nms':
                nms_parameter = nms_cfg.pop('nms_parameter')
                result_list = []
                result_list.append(
                     self._bboxes_nms(yolo_scores.squeeze(0), pred_distri_list.squeeze(0), cfg=nms_parameter))
                return result_list

            # 将原始项目的后处理转化成mmdetection的格式
            pred = preds[0]
            label = pred[:, :1].reshape(-1)
            score = pred[:, 1:2]
            bbox = pred[:, 2:]
            bboxes = torch.cat((bbox, score), -1)
            det = (bboxes, label)

            result_list = []
            result_list.append(det)

            return result_list
            # bbox_pred, bbox_num, _ = self.nms(pred_bboxes, pred_scores)
            # return bbox_pred, bbox_num


    def _bboxes_nms(self, cls_scores, bboxes, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = max_scores >= cfg['score_thr']

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg['nms'])
            return dets, labels[keep]

    def _bbox_decode(self, anchor_points, pred_dist):
        b, l, _ = get_static_shape(pred_dist)
        device = pred_dist.device
        pred_dist = pred_dist.reshape([b, l, 4, self.reg_max + 1])
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.matmul(self.proj.to(device))
        return batch_distance2bbox(anchor_points, pred_dist)

    def _bbox2distance(self, points, bbox):
        x1y1, x2y2 = torch.split(bbox, 2, -1)
        lt = points - x1y1
        rb = x2y2 - points
        return torch.cat([lt, rb], -1).clamp(0, self.reg_max - 0.01)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.int64)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float32) - target
        weight_right = 1 - weight_left

        eps = 1e-9
        # 使用混合精度训练时，pred_dist类型是torch.float16，pred_dist_act类型是torch.float32
        pred_dist_act = F.softmax(pred_dist, dim=-1)
        target_left_onehot = F.one_hot(target_left, pred_dist_act.shape[-1])
        target_right_onehot = F.one_hot(target_right, pred_dist_act.shape[-1])
        loss_left = target_left_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_right = target_right_onehot * (0 - torch.log(pred_dist_act + eps))
        loss_left = loss_left.sum(-1) * weight_left
        loss_right = loss_right.sum(-1) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _bbox_loss(self, pred_dist, pred_bboxes, anchor_points, assigned_labels,
                   assigned_bboxes, assigned_scores, assigned_scores_sum):
        # select positive samples mask
        mask_positive = (assigned_labels != self.num_classes)
        num_pos = mask_positive.sum()
        # pos/neg loss
        if num_pos > 0:
            # l1 + iou
            bbox_mask = mask_positive.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                   bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), mask_positive).unsqueeze(-1)

            loss_l1 = F.l1_loss(pred_bboxes_pos, assigned_bboxes_pos)

            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     assigned_bboxes_pos) * bbox_weight
            loss_iou = loss_iou.sum() / assigned_scores_sum

            dist_mask = mask_positive.unsqueeze(-1).repeat(
                [1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(
                pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
            assigned_ltrb = self._bbox2distance(anchor_points, assigned_bboxes)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, bbox_mask).reshape([-1, 4])
            loss_dfl = self._df_loss(pred_dist_pos,
                                     assigned_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / assigned_scores_sum
        else:
            loss_l1 = torch.zeros([]).to(pred_dist.device)
            loss_iou = torch.zeros([]).to(pred_dist.device)
            loss_dfl = pred_dist.sum() * 0.
            # loss_l1 = None
            # loss_iou = None
            # loss_dfl = None
        return loss_l1, loss_iou, loss_dfl

    @force_fp32(apply_to=('pred_scores', 'pred_distri'))
    def loss(self,
             pred_scores,
             pred_distri,
             anchors,
             anchor_points,
             num_anchors_list,
             stride_tensor,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        # 把原本forward里的操作放到这里了
        pred_scores = torch.cat(pred_scores, 1)
        pred_distri = torch.cat(pred_distri, 1)

        device = pred_scores.device
        anchors = anchors.to(device)
        anchor_points = anchor_points.to(device)
        stride_tensor = stride_tensor.to(device)

        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self._bbox_decode(anchor_points_s, pred_distri)

        gt_bbox = []
        gt_class = []
        pad_gt_mask = []
        for i in range(len(gt_bboxes)):
            gt_bbox_single = torch.zeros(1, 200, 4)
            gt_class_single = torch.zeros(1, 200, 1)
            pad_gt_mask_single = torch.zeros(1, 200, 1)
            exist_mask = torch.ones(len(gt_bboxes[i]), 1)
            gt_bbox_single[:, :len(gt_bboxes[i]), :] = gt_bboxes[i]
            pad_gt_mask_single[:, :len(gt_bboxes[i]), :] = exist_mask
            gt_bbox.append(gt_bbox_single)
            gt_class.append(gt_class_single)
            pad_gt_mask.append(pad_gt_mask_single)

        gt_bboxes = torch.cat(gt_bbox, 0)
        gt_labels = torch.cat(gt_class, 0)
        gt_labels = gt_labels.to(torch.int64)
        pad_gt_mask = torch.cat(pad_gt_mask, 0)

        # miemie2013: 剪掉填充的gt
        num_boxes = pad_gt_mask.sum([1, 2])
        num_max_boxes = num_boxes.max().to(torch.int32)
        pad_gt_mask = pad_gt_mask[:, :num_max_boxes, :]
        gt_labels = gt_labels[:, :num_max_boxes, :]
        gt_bboxes = gt_bboxes[:, :num_max_boxes, :]

        # label assignment
        if self.epoch < self.static_assigner_epoch:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.static_assigner(
                    anchors,
                    num_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    pad_gt_mask,
                    bg_index=self.num_classes,
                    pred_bboxes=pred_bboxes.detach() * stride_tensor)
            alpha_l = 0.25

        else:
            assigned_labels, assigned_bboxes, assigned_scores = \
                self.assigner(
                pred_scores.detach(),
                pred_bboxes.detach() * stride_tensor,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index=self.num_classes)
            alpha_l = -1
        # rescale bbox
        assigned_bboxes /= stride_tensor
        # cls loss
        if self.use_varifocal_loss:
            one_hot_label = F.one_hot(assigned_labels,
                                      self.num_classes + 1)[..., :-1]
            loss_cls = self._varifocal_loss(pred_scores, assigned_scores,
                                            one_hot_label)
        else:
            loss_cls = self._focal_loss(pred_scores, assigned_scores, alpha_l)

        # 每张卡上的assigned_scores_sum求平均，而且max(x, 1)
        assigned_scores_sum = assigned_scores.sum()
        assigned_scores_sum = F.relu(assigned_scores_sum - 1.) + 1.  # y = max(x, 1)
        loss_cls /= assigned_scores_sum

        loss_l1, loss_iou, loss_dfl = \
            self._bbox_loss(pred_distri, pred_bboxes, anchor_points_s,
                            assigned_labels, assigned_bboxes, assigned_scores,
                            assigned_scores_sum)

        loss_dict = dict(loss_cls=self.loss_weight['class'] * loss_cls, loss_iou=\
            self.loss_weight['iou'] * loss_iou, loss_dfl=self.loss_weight['dfl'] *loss_dfl)

        return loss_dict


    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
