# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_center(boxes):
    """Get bbox centers from boxes.
    Args:
        boxes (Tensor): boxes with shape (..., 4), "xmin, ymin, xmax, ymax" format.
    Returns:
        Tensor: boxes centers with shape (..., 2), "cx, cy" format.
    """
    boxes_cx = (boxes[..., 0] + boxes[..., 2]) / 2
    boxes_cy = (boxes[..., 1] + boxes[..., 3]) / 2
    return torch.stack([boxes_cx, boxes_cy], dim=-1)


def check_points_inside_bboxes(points,
                               bboxes,
                               center_radius_tensor=None,
                               eps=1e-9):
    r"""
    Args:
        points (Tensor, float32): shape[L, 2], "xy" format, L: num_anchors
        bboxes (Tensor, float32): shape[B, n, 4], "xmin, ymin, xmax, ymax" format
        center_radius_tensor (Tensor, float32): shape [L, 1]. Default: None.
        eps (float): Default: 1e-9
    Returns:
        is_in_bboxes (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    points = points.unsqueeze(0).unsqueeze(1)
    x, y = points.chunk(2, axis=-1)
    xmin, ymin, xmax, ymax = bboxes.unsqueeze(2).chunk(4, axis=-1)
    # check whether `points` is in `bboxes`
    l = x - xmin
    t = y - ymin
    r = xmax - x
    b = ymax - y
    delta_ltrb = torch.cat([l, t, r, b], -1)
    delta_ltrb_min, _ = delta_ltrb.min(-1)
    is_in_bboxes = (delta_ltrb_min > eps)
    if center_radius_tensor is not None:
        # check whether `points` is in `center_radius`
        center_radius_tensor = center_radius_tensor.unsqueeze(0).unsqueeze(1)
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        l = x - (cx - center_radius_tensor)
        t = y - (cy - center_radius_tensor)
        r = (cx + center_radius_tensor) - x
        b = (cy + center_radius_tensor) - y
        delta_ltrb_c = torch.cat([l, t, r, b], -1)
        delta_ltrb_c_min = delta_ltrb_c.min(-1)
        is_in_center = (delta_ltrb_c_min > eps)
        return (torch.logical_and(is_in_bboxes, is_in_center),
                torch.logical_or(is_in_bboxes, is_in_center))

    return is_in_bboxes.to(bboxes.dtype)


def compute_max_iou_anchor(ious):
    r"""
    For each anchor, find the GT with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_max_boxes = ious.shape[-2]
    max_iou_index = ious.argmax(axis=-2)
    is_max_iou = F.one_hot(max_iou_index, num_max_boxes).permute((0, 2, 1))
    return is_max_iou.to(ious.dtype)


def compute_max_iou_gt(ious):
    r"""
    For each GT, find the anchor with the largest IOU.
    Args:
        ious (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
    Returns:
        is_max_iou (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = ious.shape[-1]
    max_iou_index = ious.argmax(axis=-1)
    is_max_iou = F.one_hot(max_iou_index, num_anchors)
    return is_max_iou.to(ious.dtype)


def index_sample_2d(tensor, index):
    assert tensor.ndim == 2
    assert index.ndim == 2
    assert index.dtype == torch.int64
    d0, d1 = tensor.shape
    d2, d3 = index.shape
    assert d0 == d2
    tensor_ = tensor.reshape((-1, ))
    batch_ind = torch.arange(end=d0, dtype=index.dtype).unsqueeze(-1) * d1
    batch_ind = batch_ind.to(index.device)
    index_ = index + batch_ind
    index_ = index_.reshape((-1, ))
    out = tensor_[index_]
    out = out.reshape((d2, d3))
    return out


def gather_1d(tensor, index):
    assert index.ndim == 1
    assert index.dtype == torch.int64
    # d0, d1 = tensor.shape
    # d2, d3 = index.shape
    # assert d0 == d2
    # tensor_ = tensor.reshape((-1, ))
    # batch_ind = torch.arange(end=d0, dtype=index.dtype).unsqueeze(-1) * d1
    # index_ = index + batch_ind
    # index_ = index_.reshape((-1, ))
    # out = tensor_[index_]
    out = tensor[index]
    return out


def iou_similarity(box1, box2):
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    box1 = box1.to(torch.float32)
    box2 = box2.to(torch.float32)
    return bboxes_iou_batch(box1, box2, xyxy=True)


def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    # device = bboxes_b.device
    # print(device)
    bboxes_a = bboxes_a.to(torch.float32)
    # bboxes_a = bboxes_a.to(device)
    bboxes_b = bboxes_b.to(torch.float32)
    # print(bboxes_a.device)
    # print(bboxes_b.device)
    N = bboxes_a.shape[0]
    A = bboxes_a.shape[1]
    B = bboxes_b.shape[1]
    if xyxy:
        box_a = bboxes_a
        box_b = bboxes_b
    else:  # cxcywh格式
        box_a = torch.cat([bboxes_a[:, :, :2] - bboxes_a[:, :, 2:] * 0.5,
                           bboxes_a[:, :, :2] + bboxes_a[:, :, 2:] * 0.5], dim=-1)
        box_b = torch.cat([bboxes_b[:, :, :2] - bboxes_b[:, :, 2:] * 0.5,
                           bboxes_b[:, :, :2] + bboxes_b[:, :, 2:] * 0.5], dim=-1)

    box_a_rb = torch.reshape(box_a[:, :, 2:], (N, A, 1, 2))
    # box_a_rb = torch.tile(box_a_rb, [1, 1, B, 1])
    box_a_rb = box_a_rb.repeat(1, 1, B, 1)
    box_b_rb = torch.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    # box_b_rb = torch.tile(box_b_rb, [1, A, 1, 1])
    box_b_rb = box_b_rb.repeat(1, A, 1, 1)
    max_xy = torch.minimum(box_a_rb, box_b_rb)

    box_a_lu = torch.reshape(box_a[:, :, :2], (N, A, 1, 2))
    # box_a_lu = torch.tile(box_a_lu, [1, 1, B, 1])
    box_a_lu = box_a_lu.repeat(1, 1, B, 1)
    box_b_lu = torch.reshape(box_b[:, :, :2], (N, 1, B, 2))
    # box_b_lu = torch.tile(box_b_lu, [1, A, 1, 1])
    box_b_lu = box_b_lu.repeat(1, A, 1, 1)
    min_xy = torch.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = torch.reshape(area_a, (N, A, 1))
    # area_a = torch.tile(area_a, [1, 1, B])  # [N, A, B]
    area_a = area_a.repeat(1, 1, B)  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = torch.reshape(area_b, (N, 1, B))
    # area_b = torch.tile(area_b, [1, A, 1])  # [N, A, B]
    area_b = area_b.repeat(1, A, 1)  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]


class ATSSAssigner(nn.Module):
    """Bridging the Gap Between Anchor-based and Anchor-free Detection
     via Adaptive Training Sample Selection
    """

    def __init__(self,
                 topk=9,
                 num_classes=80,
                 force_gt_matching=False,
                 eps=1e-9):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.force_gt_matching = force_gt_matching
        self.eps = eps

    def _gather_topk_pyramid(self, gt2anchor_distances, num_anchors_list,
                             pad_gt_mask):
        pad_gt_mask = pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool)
        gt2anchor_distances_list = torch.split(
            gt2anchor_distances, num_anchors_list, -1)
        num_anchors_index = np.cumsum(num_anchors_list).tolist()
        num_anchors_index = [0, ] + num_anchors_index[:-1]
        is_in_topk_list = []
        topk_idxs_list = []
        for distances, anchors_index in zip(gt2anchor_distances_list,
                                            num_anchors_index):
            num_anchors = distances.shape[-1]
            topk_metrics, topk_idxs = torch.topk(
                distances, self.topk, dim=-1, largest=False)
            topk_idxs_list.append(topk_idxs + anchors_index)
            topk_idxs = torch.where(pad_gt_mask, topk_idxs,
                                     torch.zeros_like(topk_idxs))
            is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
            is_in_topk = torch.where(is_in_topk > 1,
                                      torch.zeros_like(is_in_topk), is_in_topk)
            is_in_topk_list.append(is_in_topk.to(gt2anchor_distances.dtype))
        is_in_topk_list = torch.cat(is_in_topk_list, -1)
        topk_idxs_list = torch.cat(topk_idxs_list, -1)
        return is_in_topk_list, topk_idxs_list

    @torch.no_grad()
    def forward(self,
                anchor_bboxes,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None,
                pred_bboxes=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        The assignment is done in following steps
        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt
        7. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            anchor_bboxes (Tensor, float32): pre-defined anchors, shape(L, 4),
                    "xmin, xmax, ymin, ymax" format
            num_anchors_list (List): num of anchors in each level
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes,
                    shape(B, n, 1), if None, then it will initialize with one_hot label
            pred_bboxes (Tensor, float32, optional): predicted bounding boxes, shape(B, L, 4)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C), if pred_bboxes is not None, then output ious
        """
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        device = anchor_bboxes.device
        gt_labels = gt_labels.to(device)
        gt_bboxes = gt_bboxes.to(device)
        pad_gt_mask = pad_gt_mask.to(device)
        # print(anchor_bboxes.device, gt_labels.device, gt_bboxes.device, pad_gt_mask.device)

        num_anchors, _ = anchor_bboxes.shape
        batch_size, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, self.num_classes])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        # 1. compute iou between gt and anchor bbox, [B, n, L]
        batch_anchor_bboxes = anchor_bboxes.unsqueeze(0).repeat([batch_size, 1, 1])
        ious = iou_similarity(gt_bboxes, batch_anchor_bboxes)

        # 2. compute center distance between all anchors and gt, [B, n, L]
        gt_centers = bbox_center(gt_bboxes.reshape([-1, 4])).unsqueeze(1)
        anchor_centers = bbox_center(anchor_bboxes)
        gt2anchor_distances = (gt_centers - anchor_centers.unsqueeze(0)) \
            .norm(2, dim=-1).reshape([batch_size, -1, num_anchors])

        # 3. on each pyramid level, selecting topk closest candidates
        # based on the center distance, [B, n, L]
        is_in_topk, topk_idxs = self._gather_topk_pyramid(
            gt2anchor_distances, num_anchors_list, pad_gt_mask)

        # 4. get corresponding iou for the these candidates, and compute the
        # mean and std, 5. set mean + std as the iou threshold
        iou_candidates = ious * is_in_topk
        aaaaaa1 = iou_candidates.reshape((-1, iou_candidates.shape[-1]))
        aaaaaa2 = topk_idxs.reshape((-1, topk_idxs.shape[-1]))
        iou_threshold = index_sample_2d(aaaaaa1, aaaaaa2)
        iou_threshold = iou_threshold.reshape([batch_size, num_max_boxes, -1])
        iou_threshold = iou_threshold.mean(dim=-1, keepdim=True) + \
                        iou_threshold.std(dim=-1, keepdim=True)
        is_in_topk = torch.where(
            iou_candidates > iou_threshold.repeat([1, 1, num_anchors]),
            is_in_topk, torch.zeros_like(is_in_topk))

        # 6. check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_centers, gt_bboxes)

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # 7. if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected.
        mask_positive_sum = mask_positive.sum(dim=-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            # when use fp16
            mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive.to(is_max_iou.dtype))
            # mask_positive = torch.where(mask_multiple_gts, is_max_iou, mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        # 8. make sure every gt_bbox matches the anchor
        if self.force_gt_matching:
            is_max_iou = compute_max_iou_gt(ious) * pad_gt_mask
            mask_max_iou = (is_max_iou.sum(-2, keepdim=True) == 1).repeat(
                [1, num_max_boxes, 1])
            mask_positive = torch.where(mask_max_iou, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        assigned_gt_index = mask_positive.argmax(-2)

        # assigned target
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(assigned_gt_index.device)
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gather_1d(
            gt_labels.flatten(), index=assigned_gt_index.flatten().to(torch.int64))
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gather_1d(
            gt_bboxes.reshape([-1, 4]), index=assigned_gt_index.flatten().to(torch.int64))
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, self.num_classes + 1)
        assigned_scores = assigned_scores.to(torch.float32)
        ind = list(range(self.num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(assigned_scores, dim=-1, index=torch.Tensor(ind).long().to(assigned_scores.device))
        if pred_bboxes is not None:
            # assigned iou
            ious = iou_similarity(gt_bboxes, pred_bboxes) * mask_positive
            ious_max, _ = ious.max(-2)
            ious_max = ious_max.unsqueeze(-1)
            assigned_scores *= ious_max
        elif gt_scores is not None:
            gather_scores = gather_1d(
                gt_scores.flatten(), index=assigned_gt_index.flatten().to(torch.int64))
            gather_scores = gather_scores.reshape([batch_size, num_anchors])
            gather_scores = torch.where(mask_positive_sum > 0, gather_scores,
                                         torch.zeros_like(gather_scores))
            assigned_scores *= gather_scores.unsqueeze(-1)
        # if torch.isnan(assigned_scores).any():
        #     print()
        return assigned_labels, assigned_bboxes, assigned_scores
