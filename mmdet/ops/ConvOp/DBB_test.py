import torch.nn as nn
import torch
from mmcv.cnn import fuse_conv_bn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from collections import OrderedDict

class DBBBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, group=1, norm_cfg=dict(type='BN', requires_grad=True)):
        super(DBBBlock, self).__init__()
        self.kernel_size = kernel_size
        self.norm_cfg = norm_cfg
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv_kxk = ConvModule(in_ch, out_ch,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=kernel_size//2,
                                   norm_cfg=norm_cfg,
                                   groups=1,
                                   act_cfg=None)
        kxk_1x1 = OrderedDict()

        kxk_1x1["conv_1x1"] = ConvModule(out_ch, out_ch, kernel_size=1, stride=1, norm_cfg=norm_cfg, groups=1, act_cfg=None)
        kxk_1x1["conv_kxk"] = ConvModule(in_ch, out_ch, kernel_size=kernel_size,
                                                        stride=stride,
                                                        padding=kernel_size//2,
                                                        norm_cfg=norm_cfg,
                                                        groups=1,
                                                        act_cfg=None)
        self.kxk_1x1 = nn.Sequential(kxk_1x1)

        conv_1x1_avg = OrderedDict()
        conv_1x1_avg["conv_1x1"] = ConvModule(out_ch, out_ch, kernel_size=1,
                                                              stride=1,
                                                              norm_cfg=norm_cfg,
                                                              groups=1,
                                                              act_cfg=None)
        conv_1x1_avg["avg"] = nn.AvgPool2d(3, padding=1, stride=1)
        self.conv_1x1_avg = nn.Sequential(conv_1x1_avg)


        self.conv_1x1 = ConvModule(in_ch, out_ch, kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                       norm_cfg=norm_cfg, act_cfg=None)
        self.stride = stride

        self.conv = nn.ModuleList([self.conv_kxk, self.kxk_1x1, self.conv_1x1_avg, self.conv_1x1])


        self.init_weight()

    def init_weight(self):
        def ini(module):
            for name, child in module.named_children():
                if isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
                    nn.init.constant_(child.running_var, 0.3)
                else:
                    if isinstance(child, nn.Module):
                        ini(child)
            return
        ini(self)
    def fuse_conv(self):
        fuse_conv_bn(self)

        conv_1x1_weight = nn.Parameter(torch.zeros_like(self.conv_kxk.conv.weight))
        # conv_1x1_bias = nn.Parameter(self.conv_kxk.conv.bia.zero_like())
        conv_1x1_weight[:, :, self.kernel_size // 2:self.kernel_size // 2 + 1,
                              self.kernel_size // 2:self.kernel_size // 2 + 1] = self.conv_1x1.conv.weight
        conv_1x1_bias = self.conv_1x1.conv.bias



        conv_1x1_kxk_weight = F.conv2d(self.kxk_1x1.conv_kxk.conv.weight, self.kxk_1x1.conv_1x1.conv.weight.permute(1, 0, 2, 3))
        conv_1x1_kxk_bias = (self.kxk_1x1.conv_kxk.conv.weight*self.kxk_1x1.conv_1x1.conv.bias.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + \
                             self.kxk_1x1.conv_kxk.conv.bias

        avg_pooling_weight = nn.Parameter(torch.ones_like(self.conv_kxk.conv.weight) / self.kernel_size**2)
        avg_pooling_mask = torch.eye(self.in_ch).reshape(self.in_ch, self.in_ch, 1, 1).to(self.conv_kxk.conv.weight.device)
        avg_pooling_weight = avg_pooling_mask * avg_pooling_weight

        conv_1x1_avg_weight = F.conv2d(avg_pooling_weight, self.conv_1x1_avg.conv_1x1.conv.weight.permute(1, 0, 2, 3))
        conv_1x1_avg_bias = (avg_pooling_weight * self.conv_1x1_avg.conv_1x1.conv.bias.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + \
                             self.kxk_1x1.conv_kxk.conv.bias


        self.conv_kxk.conv.weight = nn.Parameter(conv_1x1_weight + conv_1x1_kxk_weight + conv_1x1_avg_weight + self.conv_kxk.conv.weight)
        self.conv_kxk.conv.bias = nn.Parameter(conv_1x1_bias + conv_1x1_kxk_bias + conv_1x1_avg_bias + self.conv_kxk.conv.bias)

        self.conv = nn.ModuleList([self.conv_kxk])
    def forward(self, x):
        out = []
        for module in self.conv:
            out.append(module(x))
        res = out[0]
        for i in range(1, len(out)):
            res += out[i]
        res = F.relu(res)
        return res


x = torch.randn(1, 64, 256, 256)
DBBUnit = DBBBlock(64, 64, stride=1)
# DBBUnit.eval()
y_train = DBBUnit(x)
DBBUnit.fuse_conv()
y_test = DBBUnit(x)

print(torch.abs((y_train-y_test).sum()))
print(111)
