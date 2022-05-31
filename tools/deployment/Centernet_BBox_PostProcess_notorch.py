import onnx
from onnx_inference import ONNXModel
import onnxruntime
import cv2
import numpy as np
import time
import torch


def get_bboxes(center_heatmap_preds,
               center_heatmap_maxpool,
               wh_preds,
               offset_preds,
               img_height,
               img_weight):
    """Transform network output for a batch into bbox predictions.

    Args:
        center_heatmap_preds (list[Tensor]): center predict heatmaps for
            all levels with shape (B, num_classes, H, W).
        wh_preds (list[Tensor]): wh predicts for all levels with
            shape (B, 2, H, W).
        offset_preds (list[Tensor]): offset predicts for all levels
            with shape (B, 2, H, W).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        rescale (bool): If True, return boxes in original image space.
            Default: True.
        with_nms (bool): If True, do nms before return boxes.
            Default: False.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where 5 represent
            (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            The shape of the second tensor in the tuple is (n,), and
            each element represents the class label of the corresponding
            box.
    """
    assert len(center_heatmap_preds) == len(wh_preds) == len(
        offset_preds) == 1

    # batch_det_bboxes with shape(B,k,4+1),  coordinate and score; batch_labels with shape(B,k)
    batch_det_bboxes, batch_labels = decode_heatmap(
        center_heatmap_preds,
        center_heatmap_maxpool,
        wh_preds,
        offset_preds,
        img_height,
        img_weight)

    det_results = [
            tuple(bs) for bs in zip(batch_det_bboxes, batch_labels)
    ]
    return det_results

def decode_heatmap(center_heatmap_pred,
                   center_heatmap_maxpool,
                   wh_pred,
                   offset_pred,
                   inp_h,
                   inp_w,
                   k=100,
                   kernel=3):
    """Transform outputs into detections raw bbox prediction.

    Args:
        center_heatmap_pred (Tensor): center predict heatmap,
           shape (B, num_classes, H, W).
        wh_pred (Tensor): wh predict, shape (B, 2, H, W).
        offset_pred (Tensor): offset predict, shape (B, 2, H, W).
        img_shape (list[int]): image shape in [h, w] format.
        k (int): Get top k center keypoints from heatmap. Default 100.
        kernel (int): Max pooling kernel for extract local maximum pixels.
           Default 3.

    Returns:
        tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
           the following Tensors:

          - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
          - batch_topk_labels (Tensor): Categories of each box with \
              shape (B, k)
    """
    height, width = center_heatmap_pred.shape[2:]

    center_heatmap_pred = get_local_maximum(
        center_heatmap_pred, center_heatmap_maxpool, kernel=kernel)

    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
        center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_labels = batch_dets

    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)

    batch, top_k = topk_xs.shape
    tl_x = np.random.randn(batch, top_k)
    tl_y = np.random.randn(batch, top_k)
    br_x = np.random.randn(batch, top_k)
    br_y = np.random.randn(batch, top_k)

    batch_bboxes = []
    for i in range(batch):
        batch_ = []
        for j in range(top_k):
            topk_xs[i][j] = topk_xs[i][j] + offset[i][j][0]
            topk_ys[i][j] = topk_ys[i][j] + offset[i][j][1]
            tl_x[i][j] = (topk_xs[i][j] - wh[i][j][0] / 2) * (inp_w / width)
            tl_y[i][j] = (topk_ys[i][j] - wh[i][j][1] / 2) * (inp_h / height)
            br_x[i][j] = (topk_xs[i][j] + wh[i][j][0] / 2) * (inp_w / width)
            br_y[i][j] = (topk_ys[i][j] + wh[i][j][1] / 2) * (inp_h / height)
            batch_.append([tl_x[i][j], tl_y[i][j], br_x[i][j], br_y[i][j], batch_scores[i][j]])
        batch_bboxes.append(batch_)

    batch_bboxes = np.array(batch_bboxes)

    return batch_bboxes, batch_topk_labels

def get_local_maximum(heat, hmax, kernel=3):
    """Extract local maximum pixel with given kernal.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    batch, channel, height, width = heat.shape
    for i in range(batch):
        for j in range(channel):
            for k in range(height):
                for l in range(width):
                    if hmax[i][j][k][l] != heat[i][j][k][l]:
                        heat[i][j][k][l] = 0
    return heat

def get_topk_from_heatmap(scores, k=100):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, channel, height, width = scores.shape

    feature_number = channel*height*width
    scores_transpose = np.random.randn(batch, feature_number)
    for i in range(batch):
        for j in range(feature_number):
            scores_transpose[i][j] = scores[i][j//(height*width)][(j%(height*width))//width][(j%(height*width))%width]

    topk_scores = []
    topk_inds = []
    for i in range(batch):
        topk_scores_batch = []
        topk_inds_batch = []
        max_value_limit = 10000
        for j in range(k):
            max_value = 0
            max_index = 0
            for l in range( scores_transpose.shape[1]):
                if  scores_transpose[i][l] > max_value and  scores_transpose[i][l] < max_value_limit:
                    max_value =  scores_transpose[i][l]
                    max_index = l

            topk_scores_batch.append(max_value)
            topk_inds_batch.append(max_index)
            max_value_limit = topk_scores_batch[-1]

        topk_scores.append(topk_scores_batch)
        topk_inds.append(topk_inds_batch)

    topk_scores = np.array(topk_scores)
    topk_inds = np.array(topk_inds)

    topk_clses = np.random.randn(batch, k)
    topk_ys = np.random.randn(batch, k)
    topk_xs = np.random.randn(batch, k)

    for i in range(batch):
        for j in range(k):
            topk_clses[i][j] = topk_inds[i][j] // (height * width)
            topk_inds[i][j] = topk_inds[i][j] % (height * width)
            topk_ys[i][j] = topk_inds[i][j] // width
            topk_xs[i][j] = topk_inds[i][j] %  width

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    batch, channel, height, width = feat.shape
    feat_transpose = np.random.randn(batch, height, width, channel)
    for i in range(batch):
        for j in range(height):
            for k in range(width):
                for l in range(channel):
                    feat_transpose[i][j][k][l] = feat[i][l][j][k]

    area = height * width
    feat_transpose_reshape = np.random.randn(batch, height*width, channel)
    for i in range(batch):
        for j in range(area):
            for k in range(channel):
                feat_transpose_reshape[i][j][k] = feat_transpose[i][j//width][j%width][k]

    feat = gather_feat(feat_transpose_reshape, ind)
    return feat

def gather_feat(feat, ind):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.shape[2]
    batch, top_k = ind.shape

    feat_index = np.random.randn(batch, top_k, dim)
    for i in range(batch):
        for j in range(top_k):
            for k in range(dim):
                feat_index[i][j][k] = feat[i][int(ind[i][j])][k]

    return feat_index

def draw_box(img, p_pred):
    for i in range(p_pred.shape[0]):
        if p_pred[i][4] > 0.3:
            left_coordinate = (int(p_pred[i][0]), int(p_pred[i][1]))
            right_coordinate = (int(p_pred[i][2]), int(p_pred[i][3]))
            img = cv2.rectangle(img, left_coordinate, right_coordinate, (0, 0, 0), 3)
            img = cv2.putText(img, 'Score:{:.3f}'.format(p_pred[i][4]), left_coordinate, cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
    return img


def main():

    img_size = 320
    output_file = 'centernet_sim.onnx'
    img_path = 'test.jpg'

    # check onnx model
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)

    # print(onnx.helper.printable_graph(onnx_model.graph))
    # centernet_detector = onnxruntime.InferenceSession(output_file)
    centernet_detector = ONNXModel(output_file)

    # read BGR image
    img_read = cv2.imread(img_path)
    img_height, img_weight = img_read.shape[:2]

    # Whether BGR image needs to be converted to RGB image
    img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    # img = img_read[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    img = (img-127.5) / 128
    # add batch dimension
    img = np.expand_dims(img, 0)

    # compute ONNX Runtime output prediction
    # input_image = {"images": img}

    t1 = time.time()
    center_heatmap_pred, center_heatmap_maxpool, wh_pred, offset_pred = centernet_detector.forward(img)
    # results = centernet_detector.run(None, input_image)
    t2 = time.time()
    print(t2 - t1)

    # [tuple(img1_bboxes, img1_labels), tuple(img2_bboxes, img2_labels), ...]
    t3 = time.time()
    pred_results = get_bboxes(center_heatmap_pred,
                              center_heatmap_maxpool,
                              wh_pred,
                              offset_pred,
                              img_height,
                              img_weight)
    # Becuase B=1, p_pred with shape(k,5); label_pred with shape(k);
    t4 = time.time()
    print(t4 - t3)
    p_pred, label_pred = pred_results[0]
    img_pred = draw_box(img_read, p_pred)
    cv2.imwrite('test_pred.jpg', img_pred)


if __name__ == '__main__':
    main()