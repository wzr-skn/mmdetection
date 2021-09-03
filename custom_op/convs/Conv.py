from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.registry import CONV_LAYERS

@CONV_LAYERS.register_module()
class SepConv(ConvModule):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       dialtion=1,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')):
        assert in_channels == out_channels, "DepthWise Conv, in_channels should equal out_channels"
        super(SepConv, self).__init__(in_channels,
                                      out_channels,
                                      kernel_size,
                                      padding=kernel_size//2,
                                      groups=in_channels,
                                      dilation=dialtion,
                                      norm_cfg=norm_cfg,
                                      act_cfg=act_cfg)
    def forward(self, x, activate=True, norm=True):
        return super(SepConv, self).forward(x, activate=True, norm=True)
