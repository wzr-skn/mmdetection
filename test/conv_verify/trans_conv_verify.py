import torch
import torch.nn as nn


def transposed_conv_official():
    feature_map = torch.as_tensor([[1, 0], [2, 1]], dtype=torch.float32).reshape([1, 1, 2, 2])
    print(feature_map)
    trans_conv = nn.ConvTranspose2d(in_channels=1, out_channels=1,
                                    kernel_size=3, stride=1, bias=False)
    trans_conv.load_state_dict({"weight": torch.as_tensor([[1, 0, 1],
                                                           [0, 1, 1],
                                                           [1, 0, 0]], dtype=torch.float32).reshape([1, 1, 3, 3])})
    print(trans_conv.weight)
    output = trans_conv(feature_map)
    print(output)


def transposed_conv_self():
    feature_map = torch.as_tensor([[0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0],
                                   [0, 0, 2, 1, 0, 0],
                                   [0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0]], dtype=torch.float32).reshape([1, 1, 6, 6])
    print(feature_map)
    conv = nn.Conv2d(in_channels=1, out_channels=1,
                     kernel_size=3, stride=1, bias=False)
    conv.load_state_dict({"weight": torch.as_tensor([[0, 0, 1],
                                                    [1, 1, 0],
                                                    [1, 0, 1]], dtype=torch.float32).reshape([1, 1, 3, 3])})
    print(conv.weight)
    output = conv(feature_map)
    print(output)


def main():
    transposed_conv_official()
    print("-------------")
    transposed_conv_self()


if __name__ == '__main__':
    main()