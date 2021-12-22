from .single_stage import SingleStageDetector
from ..builder import DETECTORS
@DETECTORS.register_module()
class TTFNet(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TTFNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained)

    def forward_dummy(self, img):
        hm, wh = super().forward_dummy(img)
        import torch.nn.functional as F
        hm = F.sigmoid(hm)
        hm_maxpool = F.max_pool2d(hm, 3, stride=1, padding=1)
        return hm, hm_maxpool, wh

