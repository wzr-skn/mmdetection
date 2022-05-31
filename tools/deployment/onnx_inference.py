# -*-coding: utf-8 -*-


import os, sys
import cv2
sys.path.append(os.getcwd())
import onnxruntime
import numpy as np


class ONNXModel():
    def __init__(self, onnx_path):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_tensor):
        """
        input_feed={self.input_name: image_tensor}
        :param input_name:
        :param image_tensor:
        :return:
        """
        input_feed = {}
        for name in input_name:
            input_feed[name] = image_tensor
        return input_feed

    def forward(self, image_tensor):
        '''
        image_tensor = image.transpose(2, 0, 1)
        image_tensor = image_tensor[np.newaxis, :]
        onnx_session.run([output_name], {input_name: x})
        :param image_tensor:
        :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_tensor})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: image_tensor})
        input_feed = self.get_input_feed(self.input_name, image_tensor)
        boxes = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return boxes

if __name__ == "__main__":
    output_file = './tools/deployment/generalsandnet_final.onnx'
    img_path = './tools/deployment/4.jpg'
    detector = ONNXModel(output_file)

    img_read = cv2.imread(img_path)
    img_read = cv2.resize(img_read, (128, 128))
    # Whether BGR image needs to be converted to RGB image
    img = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
    # img = img_read[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    img = (img-127.5) / 128
    # add batch dimension
    img = np.expand_dims(img, 0)

    res = detector.forward(img)