import torch
from onnx_inference import ONNXModel
from argparse import ArgumentParser
import os
import cv2
import numpy as np
from mmdet.apis import init_detector
import time


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('onnx_path', help='onnx model path')
    parser.add_argument('pth_path', help='pytorch model path')
    parser.add_argument('img_file', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    args = parser.parse_args()
    return args


def main(args):
    # build the model from a config file and a checkpoint file
    onnx_model = ONNXModel(args.onnx_path)
    pth_model = init_detector(args.config, args.pth_path, device=args.device)
    # test a single image
    img_file_list = os.listdir(args.img_file)
    for img_name in img_file_list:
        print(img_name)
        img_path = os.path.join(args.img_file, img_name)
        img_read = cv2.imread(img_path)
        img_read = cv2.resize(img_read, (480, 480))
        # Whether BGR image needs to be converted to RGB image
        img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        # img = img_read[:, :, ::-1]
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = (img - 127.5) / 128
        # add batch dimension
        img = np.expand_dims(img, 0)
        img_torch = torch.from_numpy(img)
        img_torch = img_torch.to(args.device)
        t1 = time.time()
        onnx_result = onnx_model.forward(img)
        pth_result = pth_model.forward_dummy(img_torch)
        t2 = time.time()
        print(t2-t1)
        for i in range(len(onnx_result)):
            print(onnx_result[i].sum())
            print(pth_result[int(i/3)][i%3].sum())
            print(onnx_result[i].max())
            print(pth_result[int(i/3)][i%3].max())


if __name__ == '__main__':
    args = parse_args()
    main(args)
