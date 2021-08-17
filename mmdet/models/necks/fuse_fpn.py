import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.ops.build import build_op
from mmcv.runner import auto_fp16
from ..builder import NECKS
from mmcv.cnn import ConvModule, xavier_init

@NECKS.register_module()
class FuseFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stack_conv = 2,
                 conv_type="NormalConv",
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(FuseFPN, self).__init__()
        # add extra bottom up pathway
        num_in = len(in_channels)
        self.in_channels = in_channels
        self.upsample_conv = nn.ModuleList()
        self.lateral_conv = nn.ModuleList()

        Conv = build_op(conv_type)

        for i in range(num_in):
            lateral_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                padding=0,
                norm_cfg=norm_cfg,
                inplace=True)
            if i < num_in - 1:
                upsample_conv = nn.Sequential()
                for index in range(stack_conv):
                    upsample_conv.add_module(name="upsample_conv3x3_{}".format(index),
                                             module=Conv(
                                                    out_channels,
                                                    out_channels,
                                                    3,
                                                    norm_cfg=norm_cfg))
                    upsample_conv.add_module(name="upsample_conv1x1_{}".format(index),
                                             module=ConvModule(
                                                    out_channels,
                                                    out_channels,
                                                    1,
                                                    stride=1,
                                                    padding=0,
                                                    norm_cfg=norm_cfg,
                                                    inplace=True))
                self.upsample_conv.append(upsample_conv)
            self.lateral_conv.append(lateral_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build laterals

        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_conv)
        ]

        # build top-down path
        for i in range(len(inputs)-1, 0, -1):
            laterals[i-1] = self.upsample_conv[i-1](F.interpolate(laterals[i], laterals[i-1].shape[2:]) + laterals[i-1])


        # build outputs
        # part 1: from original levels

        return tuple([laterals[0]])

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
