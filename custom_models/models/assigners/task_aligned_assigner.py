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

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def gather_nd(tensor, index):
    if tensor.ndim == 4 and index.ndim == 2:
        N, R, S, T = tensor.shape
        index_0 = index[:, 0]  # [M, ]
        index_1 = index[:, 1]  # [M, ]
        index_2 = index[:, 2]  # [M, ]
        index_ = index_0 * R * S + index_1 * S + index_2  # [M, ]
        x2 = torch.reshape(tensor, (N * R * S, T))  # [N*R*S, T]
        index_ = index_.to(torch.int64)
        out = gather_1d(x2, index_)
    elif tensor.ndim == 3 and index.ndim == 3:
        A, B, C = tensor.shape
        D, E, F = index.shape
        assert F == 2
        # out.shape = [D, E, C]
        tensor_ = tensor.reshape((-1, C))   # [A*B, C]
        index_ = index.reshape((-1, F))     # [D*E, F]


        index_0 = index_[:, 0]  # [D*E, ]
        index_1 = index_[:, 1]  # [D*E, ]
        index_ = index_0 * B + index_1  # [D*E, ]

        out = gather_1d(tensor_, index_)  # [D*E, C]
        out = out.reshape((D, E, C))   # [D, E, C]
    else:
        raise NotImplementedError("not implemented.")
    return out


def bboxes_iou_batch(bboxes_a, bboxes_b, xyxy=True):
    """计算两组矩形两两之间的iou
    Args:
        bboxes_a: (tensor) bounding boxes, Shape: [N, A, 4].
        bboxes_b: (tensor) bounding boxes, Shape: [N, B, 4].
    Return:
      (tensor) iou, Shape: [N, A, B].
    """
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    bboxes_a = bboxes_a.to(torch.float32)
    bboxes_b = bboxes_b.to(torch.float32)
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
    box_a_rb = torch.tile(box_a_rb, [1, 1, B, 1])
    # box_a_rb = box_a_rb.repeat(1, 1, B, 1)
    box_b_rb = torch.reshape(box_b[:, :, 2:], (N, 1, B, 2))
    box_b_rb = torch.tile(box_b_rb, [1, A, 1, 1])
    # box_b_rb = box_b_rb.repeat(1, A, 1, 1)
    max_xy = torch.minimum(box_a_rb, box_b_rb)

    box_a_lu = torch.reshape(box_a[:, :, :2], (N, A, 1, 2))
    box_a_lu = torch.tile(box_a_lu, [1, 1, B, 1])
    # box_a_lu = box_a_lu.repeat(1, 1, B, 1)
    box_b_lu = torch.reshape(box_b[:, :, :2], (N, 1, B, 2))
    box_b_lu = torch.tile(box_b_lu, [1, A, 1, 1])
    # box_b_lu = box_b_lu.repeat(1, A, 1, 1)
    min_xy = torch.maximum(box_a_lu, box_b_lu)

    inter = F.relu(max_xy - min_xy)
    inter = inter[:, :, :, 0] * inter[:, :, :, 1]

    box_a_w = box_a[:, :, 2]-box_a[:, :, 0]
    box_a_h = box_a[:, :, 3]-box_a[:, :, 1]
    area_a = box_a_h * box_a_w
    area_a = torch.reshape(area_a, (N, A, 1))
    area_a = torch.tile(area_a, [1, 1, B])  # [N, A, B]
    # area_a = area_a.repeat(1, 1, B)  # [N, A, B]

    box_b_w = box_b[:, :, 2]-box_b[:, :, 0]
    box_b_h = box_b[:, :, 3]-box_b[:, :, 1]
    area_b = box_b_h * box_b_w
    area_b = torch.reshape(area_b, (N, 1, B))
    area_b = torch.tile(area_b, [1, A, 1])  # [N, A, B]
    # area_b = area_b.repeat(1, A, 1)  # [N, A, B]

    union = area_a + area_b - inter + 1e-9
    return inter / union  # [N, A, B]

def iou_similarity(box1, box2):
    # 使用混合精度训练时，iou可能出现nan，所以转成torch.float32
    box1 = box1.to(torch.float32)
    box2 = box2.to(torch.float32)
    return bboxes_iou_batch(box1, box2, xyxy=True)


def gather_topk_anchors(metrics, topk, largest=True, topk_mask=None, eps=1e-9):
    r"""
    Args:
        metrics (Tensor, float32): shape[B, n, L], n: num_gts, L: num_anchors
        topk (int): The number of top elements to look for along the axis.
        largest (bool) : largest is a flag, if set to true,
            algorithm will sort by descending order, otherwise sort by
            ascending order. Default: True
        topk_mask (Tensor, bool|None): shape[B, n, topk], ignore bbox mask,
            Default: None
        eps (float): Default: 1e-9
    Returns:
        is_in_topk (Tensor, float32): shape[B, n, L], value=1. means selected
    """
    num_anchors = metrics.shape[-1]
    topk_metrics, topk_idxs = torch.topk(
        metrics, topk, dim=-1, largest=largest)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).tile(
            [1, 1, topk])
        # topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > eps).repeat(
        #     1, 1, topk)
    topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
    is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
    is_in_topk = torch.where(is_in_topk > 1,
                              torch.zeros_like(is_in_topk), is_in_topk)
    return is_in_topk.to(metrics.dtype)


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


class TaskAlignedAssigner(nn.Module):
    """TOOD: Task-aligned One-stage Object Detection
    """

    def __init__(self, topk=13, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self,
                pred_scores,
                pred_bboxes,
                anchor_points,
                num_anchors_list,
                gt_labels,
                gt_bboxes,
                pad_gt_mask,
                bg_index,
                gt_scores=None):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/task_aligned_assigner.py

        The assignment is done in following steps
        1. compute alignment metric between all bbox (bbox of all pyramid levels) and gt
        2. select top-k bbox as candidates for each gt
        3. limit the positive sample's center in gt (because the anchor-free detector
           only can predict positive distance)
        4. if an anchor box is assigned to multiple gts, the one with the
           highest iou will be selected.
        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 4)
            anchor_points (Tensor, float32): pre-defined anchors, shape(L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 4)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 4)
            assigned_scores (Tensor): (B, L, C)
        """
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3

        device = pred_scores.device
        pred_bboxes = pred_bboxes.to(device)
        gt_labels = gt_labels.to(device)
        gt_bboxes = gt_bboxes.to(device)
        pad_gt_mask = pad_gt_mask.to(device)
        print(pred_scores.device, pred_bboxes.device, gt_labels.device, gt_bboxes.device, pad_gt_mask.device)

        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape

        # negative batch
        if num_max_boxes == 0:
            assigned_labels = torch.full(
                [batch_size, num_anchors], bg_index, dtype=gt_labels.dtype)
            assigned_bboxes = torch.zeros([batch_size, num_anchors, 4])
            assigned_scores = torch.zeros(
                [batch_size, num_anchors, num_classes])

            assigned_labels = assigned_labels.to(gt_bboxes.device)
            assigned_bboxes = assigned_bboxes.to(gt_bboxes.device)
            assigned_scores = assigned_scores.to(gt_bboxes.device)
            return assigned_labels, assigned_bboxes, assigned_scores

        # compute iou between gt and pred bbox, [B, n, L]
        ious = iou_similarity(gt_bboxes, pred_bboxes)
        # gather pred bboxes class score
        pred_scores = pred_scores.permute([0, 2, 1])
        batch_ind = torch.arange(
            end=batch_size, dtype=gt_labels.dtype).unsqueeze(-1)
        batch_ind = batch_ind.to(gt_labels.device)
        gt_labels_ind = torch.stack([batch_ind.repeat([1, num_max_boxes]), gt_labels.squeeze(-1)], -1)
        bbox_cls_scores = gather_nd(pred_scores, gt_labels_ind)
        # compute alignment metrics, [B, n, L]
        alignment_metrics = bbox_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        # check the positive sample's center in gt, [B, n, L]
        is_in_gts = check_points_inside_bboxes(anchor_points, gt_bboxes)

        # select topk largest alignment metrics pred bbox as candidates
        # for each gt, [B, n, L]
        is_in_topk = gather_topk_anchors(
            alignment_metrics * is_in_gts,
            self.topk,
            topk_mask=pad_gt_mask.repeat([1, 1, self.topk]).to(torch.bool))

        # select positive sample, [B, n, L]
        mask_positive = is_in_topk * is_in_gts * pad_gt_mask

        # if an anchor box is assigned to multiple gts,
        # the one with the highest iou will be selected, [B, n, L]
        mask_positive_sum = mask_positive.sum(-2)
        if mask_positive_sum.max() > 1:
            mask_multiple_gts = (mask_positive_sum.unsqueeze(1) > 1).repeat(
                [1, num_max_boxes, 1])
            is_max_iou = compute_max_iou_anchor(ious)
            mask_positive = torch.where(mask_multiple_gts, is_max_iou,
                                         mask_positive)
            mask_positive_sum = mask_positive.sum(-2)
        assigned_gt_index = mask_positive.argmax(-2)

        # assigned target
        assigned_gt_index = assigned_gt_index + batch_ind * num_max_boxes
        assigned_labels = gather_1d(gt_labels.flatten(), assigned_gt_index.flatten())
        assigned_labels = assigned_labels.reshape([batch_size, num_anchors])
        assigned_labels = torch.where(
            mask_positive_sum > 0, assigned_labels,
            torch.full_like(assigned_labels, bg_index))

        assigned_bboxes = gather_1d(
            gt_bboxes.reshape([-1, 4]), assigned_gt_index.flatten())
        assigned_bboxes = assigned_bboxes.reshape([batch_size, num_anchors, 4])

        assigned_scores = F.one_hot(assigned_labels, num_classes + 1)
        ind = list(range(num_classes + 1))
        ind.remove(bg_index)
        assigned_scores = torch.index_select(
            assigned_scores, dim=-1, index=torch.Tensor(ind).long().to(assigned_scores.device))
        # rescale alignment metrics
        alignment_metrics *= mask_positive
        max_metrics_per_instance, _ = alignment_metrics.max(-1, keepdim=True)
        max_ious_per_instance, _ = (ious * mask_positive).max(-1, keepdim=True)
        alignment_metrics = alignment_metrics / (
            max_metrics_per_instance + self.eps) * max_ious_per_instance
        alignment_metrics, _ = alignment_metrics.max(-2)
        alignment_metrics = alignment_metrics.unsqueeze(-1)
        assigned_scores = assigned_scores * alignment_metrics

        return assigned_labels, assigned_bboxes, assigned_scores
