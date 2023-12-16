from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import CONV_LAYERS
import torch.nn as nn


@CONV_LAYERS.register_module()
class DepthWiseConv(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 bias=False):
        assert in_channels == out_channels, "DepthWise Conv, in_channels should equal out_channels"
        super(DepthWiseConv, self).__init__(in_channels, out_channels,
                 kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, groups=1, bias=bias)
        self.weight = None
        self.bias = None
        del self.weight
        del self.bias

        self.conv = nn.Sequential(
            ConvModule(in_channels=in_channels, out_channels=out_channels, groups=in_channels,
                       kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, act_cfg=None),
        )

    def forward(self, x):
        return self.conv(x)
