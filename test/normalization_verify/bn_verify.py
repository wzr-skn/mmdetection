import numpy as np
import torch.nn as nn
import torch


def bn_process(feature, mean, var):
    feature_shape = feature.shape
    for i in range(feature_shape[1]):
        #[N, C, H, W]
        feature_channel = feature[:, i, ...]
        mean_channel = feature_channel.mean()
        # 总体标准差
        std_channel_total = feature_channel.std()
        # 样本标准差
        std_channel_sample = feature_channel.std(ddof=1)

        # bn process
        feature[:, i, ...] = (feature[:, i, ...] - mean_channel) / np.sqrt(std_channel_total ** 2 + 1e-5)
        # update calculating mean and var
        mean[i] = mean[i] * 0.9 + mean_channel * 0.1
        var[i] = var[i] * 0.9 + (std_channel_sample ** 2) * 0.1
    print(feature)


if __name__ == '__main__':
    # generate feature with [batch, channel, height, width]
    feature1 = torch.randn(2, 2, 2, 2)
    # 初始化统计均值和方差
    calcalate_mean = [0, 0]
    calculate_var = [1.0, 1.0]
    # print(feature1.numpy())

    # 使用copy()浅拷贝
    bn_process(feature1.numpy().copy(), calcalate_mean, calculate_var)

    bn = nn.BatchNorm2d(2, eps=1e-5)
    output = bn(feature1)
    print(output)
