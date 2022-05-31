from argparse import ArgumentParser
import os
from mmdet.apis import init_detector
import warnings
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('out_file', help='output file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    args = parser.parse_args()
    return args

def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        img = data['img'][0]
        results = model(img)

    # 目前这样的写法适用于yolox算法，其他算法取数据后存为np.array的过程需修改
    npy_results = []
    for i in range(len(results)):
        if len(results[0]) > 0:
            for j in range(len(results[0])):
                npy_results.append(results[i][j])
        else:
            npy_results.append(results[i])

    npy_results = np.array(npy_results)

    # cls1 = results[0][0].detach().numpy().ravel()
    # reg1 = results[1][0].detach().numpy().ravel()
    # obj1 = results[2][0].detach().numpy().ravel()
    # npy_results = np.array([cls1, cls2, cls3, reg1, reg2, reg3, obj1, obj2, obj3])
    # for arr in npy_results:
    #     arr = arr.reshape(2,-1)

    return npy_results

def main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    forward_dummy = model.forward_dummy
    model.forward = forward_dummy

    # test a single image
    assert os.path.isdir(args.out_file)
    img_file_list = os.listdir(args.img_file)
    for img_name in img_file_list:
        if img_name[-3:] not in ["jpg", "png", "bmp"]:
            warnings.warn(f"{img_name} is not a image name")
            continue
        print(img_name)
        img_path = os.path.join(args.img_file, img_name)
        out_path = os.path.join(args.out_file, 'batch_{}.npy'.format(img_name[:-4]))

        result = inference_detector(model, img_path)
        np.save(out_path, result)


if __name__ == '__main__':
    args = parse_args()
    main(args)
