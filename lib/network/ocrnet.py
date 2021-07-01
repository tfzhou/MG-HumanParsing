import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.module_helper import ModuleHelper
from .modules.spatial_ocr_block import SpatialOCR_Module
from .modules.spatial_ocr_block import SpatialGather_Module

from .hrnet_package.hrnet_backbone import HRNetBackbone
from .hrnet_package.hrnet_backbone_pp import HRNetBackbone as HRNetBackbonePP


class HRNet_W48_OCR(nn.Module):
    def __init__(self, pretrained, bn_type, backbone='hrnet48', num_classes=20):
        super(HRNet_W48_OCR, self).__init__()
        self.num_classes = num_classes
        self.backbone = backbone

        if backbone == 'hrnet48++':
            hrnet_backbone_func = HRNetBackbonePP(pretrained=pretrained, bn_type=bn_type, backbone=backbone)
        else:
            hrnet_backbone_func = HRNetBackbone(pretrained=pretrained, bn_type=bn_type, backbone=backbone)
        self.backbone = hrnet_backbone_func()

        if backbone == 'hrnet48':
            in_channels = 720
        elif backbone == 'hrnet48++':
            in_channels = 2944

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=bn_type))
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=512,
                                                 key_channels=256,
                                                 out_channels=512,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type=bn_type)
        self.cls_head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        if backbone == 'hrnet48':
            self.aux_head = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(in_channels, bn_type=bn_type),
                nn.Conv2d(in_channels, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
                )
        else:
            self.aux_head = nn.Sequential(
                nn.Conv2d(in_channels, 720, kernel_size=3, stride=1, padding=1),
                ModuleHelper.BNReLU(720, bn_type=bn_type),
                nn.Conv2d(720, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        out_aux = self.aux_head(feats)

        feats = self.conv3x3(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux = F.interpolate(out_aux, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        out = F.interpolate(out, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return out, out_aux


def get_model(pretrained=None, bn_type='inplace_abn', backbone='hrnet48', num_classes=20):
    model = HRNet_W48_OCR(pretrained, bn_type, num_classes=num_classes, backbone=backbone)
    return model
