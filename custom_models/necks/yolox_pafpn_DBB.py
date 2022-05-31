# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmcv.runner import BaseModule

from ..builder import NECKS
from ..backbones.csp_darknet_no import CSPLayer
from mmdet.ops.build import build_op


@NECKS.register_module()
class YOLOXPAFPN_DBB(BaseModule):
    """Path Aggregation Network used in YOLOX.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='Swish')
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_csp_blocks=3,
                 use_depthwise=False,
                 upsample_cfg=dict(scale_factor=2, mode='nearest'),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                 conv_type="DBBBlock",
                 dilate = 1,
                 only_one_output = False):
        super(YOLOXPAFPN_DBB, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.only_one_output = only_one_output

        conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        CONV_TYPE = build_op(conv_type)

        # build top-down blocks
        self.upsample = nn.Upsample(**upsample_cfg)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(
                    in_channels[idx],
                    in_channels[idx - 1],
                    1,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            # 加入了dilate配合forward部分add的修改
            self.top_down_blocks.append(
                CSPLayer(
                    in_channels[idx - 1] * dilate,
                    in_channels[idx - 1],
                    num_blocks=num_csp_blocks,
                    add_identity=False,
                    use_depthwise=use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    conv_type="DBBBlock"))

        # build bottom-up blocks
        if self.only_one_output is False:
            self.downsamples = nn.ModuleList()
            self.bottom_up_blocks = nn.ModuleList()
            for idx in range(len(in_channels) - 1):
                self.downsamples.append(
                    CONV_TYPE(
                        in_channels[idx],
                        in_channels[idx],
                        3,
                        stride=2,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
                # 加入了dilate配合forward部分add的修改
                self.bottom_up_blocks.append(
                    CSPLayer(
                        in_channels[idx] * dilate,
                        in_channels[idx + 1],
                        num_blocks=num_csp_blocks,
                        add_identity=False,
                        use_depthwise=use_depthwise,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        conv_type="DBBBlock"))

            self.out_convs = nn.ModuleList()
            for i in range(len(in_channels)):
                self.out_convs.append(
                    ConvModule(
                        in_channels[i],
                        out_channels,
                        1,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))

        else:
            self.out_convs = ConvModule(in_channels[0],
                                        out_channels,
                                        1,
                                        norm_cfg=norm_cfg,
                                        act_cfg=act_cfg)

    def forward(self, inputs):
        """
        Args:
            inputs (tuple[Tensor]): input features.

        Returns:
            tuple[Tensor]: YOLOXPAFPN features.
        """
        assert len(inputs) == len(self.in_channels)

        # top-down path
        inner_outs = [inputs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_heigh = self.reduce_layers[len(self.in_channels) - 1 - idx](
                feat_heigh)
            inner_outs[0] = feat_heigh

            upsample_feat = self.upsample(feat_heigh)

            # torch.cat被修改为了torch.add
            inner_out = self.top_down_blocks[len(self.in_channels) - 1 - idx](
                torch.add(upsample_feat, feat_low))
            inner_outs.insert(0, inner_out)

        if self.only_one_output is False:
            # bottom-up path
            outs = [inner_outs[0]]
            for idx in range(len(self.in_channels) - 1):
                feat_low = outs[-1]
                feat_height = inner_outs[idx + 1]
                downsample_feat = self.downsamples[idx](feat_low)

                # torch.cat被修改为了torch.add
                out = self.bottom_up_blocks[idx](
                    torch.add(downsample_feat, feat_height))
                outs.append(out)

            # out convs
            for idx, conv in enumerate(self.out_convs):
                outs[idx] = conv(outs[idx])

            return tuple(outs)

        else:
            outs = [self.out_convs(inner_outs[0])]
            return tuple(outs)


