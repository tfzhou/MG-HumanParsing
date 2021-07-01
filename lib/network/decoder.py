import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from functools import partial
from typing import Any, List, Tuple, Union
from .heads import IntensityMeta, AssociationMeta, DetectionMeta, ParsingMeta

from .modules.spatial_ocr_block import SpatialOCR_Module
from .modules.spatial_ocr_block import SpatialGather_Module

from .aspp import ASPP
from .conv_module import stacked_conv

import numpy as np


def conv3x3_bn_relu(in_planes, out_planes, stride=1, norm_type="batchnorm"):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True)
    )


class AlignModule(nn.Module):
    def __init__(self, inplane, outplane):
        super(AlignModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=3, padding=1, bias=False)

    def forward(self, low_feature, h_feature, pred=None):
        #low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.interpolate(h_feature,size=size, mode="bilinear", align_corners=False)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        if pred is not None:
            pred = 2 * self.flow_warp(pred, flow, size=size)
            return h_feature, pred

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class CascadeRefinementHeadXception(nn.Module):
    def __init__(self, num_classes, class_key, head_channels,
                 meta: Union[IntensityMeta, AssociationMeta, DetectionMeta, ParsingMeta]):
        super(CascadeRefinementHeadXception, self).__init__()
        self.meta = meta
        self.num_head = len(num_classes)
        self.class_key = class_key
        self.num_classes = num_classes

        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1,
                            padding=2, conv_type='depthwise_separable_conv')

        dim = 256

        # feature fusion
        self.res5_fuse = HRNetFusionModule(in_channels=2048,
                                           output_channels=dim)
        self.res4_fuse = HRNetFusionModule(in_channels=728,
                                           output_channels=dim)
        self.res3_fuse = HRNetFusionModule(in_channels=728,
                                           output_channels=dim)
        self.res2_fuse = HRNetFusionModule(in_channels=256,
                                           output_channels=dim)

        self.align_5_4 = AlignModule(dim, dim//2)
        self.align_4_3 = AlignModule(dim, dim//2)
        self.align_3_2 = AlignModule(dim, dim//2)

        self.classifier_5 = nn.Conv2d(dim, 2, kernel_size=1)
        self.classifier_4 = nn.Conv2d(dim, 2, kernel_size=1)
        self.classifier_3 = nn.Conv2d(dim, 2, kernel_size=1)
        self.classifier_2 = nn.Conv2d(dim, 2, kernel_size=1)

    def stride(self, basenet_stride):
        return basenet_stride // 4

    def forward(self, features):
        res2 = features['res2']
        res3 = features['res3']
        res4 = features['res4']
        res5 = features['res5']

        res5_feature = self.res5_fuse(res5)
        res4_feature = self.res4_fuse(res4)
        res3_feature = self.res3_fuse(res3)
        res2_feature = self.res2_fuse(res2)

        # ======== initial classifier
        pred = OrderedDict()
        pred['offset-5'] = self.classifier_5(res5_feature)

        # ======== 2nd stage
        # warp feature and initial prediction
        res4_feature, warped_pred = self.align_5_4(res4_feature, res5_feature,
                                                   pred['offset-5'])

        pred['offset-4'] = self.classifier_4(res4_feature) + warped_pred

        # ======== 3rd stage
        res3_feature, warped_pred = self.align_4_3(res3_feature, res4_feature,
                                                   pred['offset-4'])

        pred['offset-3'] = self.classifier_3(res3_feature) + warped_pred

        # ======== 4th stage
        res2_feature, warped_pred = self.align_3_2(res2_feature, res3_feature,
                                                   pred['offset-3'])

        pred['offset-2'] = self.classifier_2(res2_feature) + warped_pred

        return pred

    def voting(self, semantic, offset):
        bs = semantic.shape[0]
        hh, ww = semantic.shape[2], semantic.shape[3]

        vote_map = torch.zeros_like(offset[:, :1, ...])

        device = semantic.get_device()
        xx, yy = torch.Tensor(list(range(ww))), torch.Tensor(list(range(hh)))
        xx, yy = xx.to(device), yy.to(device)
        grid_y, grid_x = torch.meshgrid(yy, xx)

        for ii in range(bs):
            segm_pred = torch.argmax(semantic[ii, ...], axis=0)
            segm_pred[segm_pred > 0] = 1
            segm_pred = segm_pred.bool()

            coord_x = torch.masked_select(grid_x, segm_pred).long()
            coord_y = torch.masked_select(grid_y, segm_pred).long()

            offset_x = offset[ii, 1, coord_y, coord_x]
            offset_y = offset[ii, 0, coord_y, coord_x]

            vote_x = coord_x + offset_x
            vote_y = coord_y + offset_y
            vote_x = torch.clamp(vote_x, 0, ww - 1).long()
            vote_y = torch.clamp(vote_y, 0, hh - 1).long()

            indexs = vote_y * ww + vote_x
            indexs = indexs.cpu().numpy()
            counts = np.bincount(indexs, minlength=hh * ww)
            counts = counts / (np.max(counts) + 1e-10)
            counts = np.reshape(counts, (hh, ww))
            counts = torch.from_numpy(counts).float().to(device)
            vote_map[ii, 0, ...] = counts

        return vote_map


class HRNetFusionModule(nn.Module):
    def __init__(self, in_channels, output_channels,
                 astrous_rates=None, aspp_channels=None):
        super(HRNetFusionModule, self).__init__()

        if aspp_channels is not None and astrous_rates is not None:
            self.aspp = ASPP(in_channels, out_channels=aspp_channels,
                             atrous_rates=astrous_rates)
            in_channels = aspp_channels
        else:
            self.aspp = None

        self.project = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=output_channels,
                      kernel_size=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, down=None):
        h, w = x.shape[2], x.shape[3]

        if self.aspp is not None:
            x = self.aspp(x)

        if down is not None:
            down = F.interpolate(down, size=(h, w), mode='bilinear',
                                 align_corners=True)
            x = torch.cat([x, down], dim=1)

        x = self.project(x)

        return x


class OCR(nn.Module):
    def __init__(self, num_classes, in_channels, key_channels, out_channels):
        super(OCR, self).__init__()

        self.num_classes = num_classes
        self.ocr_gather_head = SpatialGather_Module(self.num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=in_channels,
                                                 key_channels=key_channels,
                                                 out_channels=out_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 bn_type='torchbn')

    def forward(self, feat, prob):
        h, w = feat.shape[2], feat.shape[3]

        prob = torch.nn.functional.upsample_bilinear(feat, (h, w))

        context = self.ocr_gather_head(feat, prob)
        feat = self.ocr_distri_head(feat, context)

        return feat


class CascadeRefinementHead(nn.Module):
    def __init__(self, num_classes, class_key, head_channels,
                 meta: Union[IntensityMeta, AssociationMeta, DetectionMeta, ParsingMeta]):
        super(CascadeRefinementHead, self).__init__()
        self.meta = meta
        self.num_head = len(num_classes)
        self.class_key = class_key
        self.num_classes = num_classes

        hrnet_channels = [48, 96, 192, 384]

        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1,
                            padding=2, conv_type='depthwise_separable_conv')

        # feature fusion
        self.res5_fuse = HRNetFusionModule(in_channels=384,
                                           astrous_rates=(3, 6, 9),
                                           aspp_channels=384,
                                           output_channels=384)
        self.res4_fuse = HRNetFusionModule(in_channels=192 + 384,
                                           output_channels=192)
        self.res3_fuse = HRNetFusionModule(in_channels=96 + 192,
                                           output_channels=96)
        self.res2_fuse = HRNetFusionModule(in_channels=48 + 96,
                                           output_channels=48)

        # feature modulation
        self.res4_ocr = OCR(self.num_classes, in_channels=192,
                            key_channels=128, out_channels=192)

        self.res3_ocr = OCR(self.num_classes, in_channels=96,
                            key_channels=128, out_channels=96)

        self.res2_ocr = OCR(self.num_classes, in_channels=48,
                            key_channels=128, out_channels=48)

        # classifier
        classifiers = {}
        for j in range(2, 6):
            for i in range(self.num_head):
                if class_key[i] == 'semantic' or j == 5:
                    in_channels = hrnet_channels[j - 2]
                else:
                    in_channels = hrnet_channels[j - 2] + 2  # with offset channels

                classifiers[class_key[i] + '-{}'.format(j)] = \
                    nn.Sequential(
                        fuse_conv(in_channels, head_channels),
                        nn.Conv2d(head_channels, num_classes[i], 1)
                    )

        self.classifiers = nn.ModuleDict(classifiers)

    def stride(self, basenet_stride):
        return basenet_stride // 4

    def forward(self, features):
        res2 = features['res2']
        res3 = features['res3']
        res4 = features['res4']
        res5 = features['res5']

        pred = OrderedDict()

        res5_feature = self.res5_fuse(res5)
        for key in self.class_key:
            pred[key + '-5'] = self.classifiers[key + '-5'](res5_feature)

        res4_feature = self.res4_fuse(res4, res5_feature)
        res4_feature = self.res4_ocr(res4_feature, pred['semantic-5'])
        for key in self.class_key:
            if key == 'semantic':
                pred[key + '-4'] = self.classifiers[key + '-4'](res4_feature)
            else:
                offset_pred = F.interpolate(pred['offset-5'],
                                            size=res4_feature.shape[2:4],
                                            mode='bilinear')
                offset_pred *= 2
                residual = self.classifiers[key + '-4'](torch.cat([res4_feature, offset_pred], dim=1))
                pred[key + '-4'] = offset_pred + residual

        res3_feature = self.res3_fuse(res3, res4_feature)
        res3_feature = self.res3_ocr(res3_feature, pred['semantic-4'])
        for key in self.class_key:
            if key == 'semantic':
                pred[key + '-3'] = self.classifiers[key + '-3'](res3_feature)
            else:
                offset_pred = F.interpolate(pred['offset-4'],
                                            size=res3_feature.shape[2:4],
                                            mode='bilinear')
                offset_pred *= 2
                residual = self.classifiers[key + '-3'](torch.cat([res3_feature, offset_pred], dim=1))
                pred[key + '-3'] = offset_pred + residual

        res2_feature = self.res2_fuse(res2, res3_feature)
        res2_feature = self.res2_ocr(res2_feature, pred['semantic-3'])
        for key in self.class_key:
            if key == 'semantic':
                pred[key + '-2'] = self.classifiers[key + '-2'](res2_feature)
            else:
                offset_pred = F.interpolate(pred['offset-3'],
                                            size=res2_feature.shape[2:4],
                                            mode='bilinear')
                offset_pred *= 2
                residual = self.classifiers[key + '-2'](torch.cat([res2_feature, offset_pred], dim=1))
                pred[key + '-2'] = offset_pred + residual

        return pred

    def voting(self, semantic, offset):
        bs = semantic.shape[0]
        hh, ww = semantic.shape[2], semantic.shape[3]

        vote_map = torch.zeros_like(offset[:, :1, ...])

        device = semantic.get_device()
        xx, yy = torch.Tensor(list(range(ww))), torch.Tensor(list(range(hh)))
        xx, yy = xx.to(device), yy.to(device)
        grid_y, grid_x = torch.meshgrid(yy, xx)

        for ii in range(bs):
            segm_pred = torch.argmax(semantic[ii, ...], axis=0)
            segm_pred[segm_pred > 0] = 1
            segm_pred = segm_pred.bool()

            coord_x = torch.masked_select(grid_x, segm_pred).long()
            coord_y = torch.masked_select(grid_y, segm_pred).long()

            offset_x = offset[ii, 1, coord_y, coord_x]
            offset_y = offset[ii, 0, coord_y, coord_x]

            vote_x = coord_x + offset_x
            vote_y = coord_y + offset_y
            vote_x = torch.clamp(vote_x, 0, ww - 1).long()
            vote_y = torch.clamp(vote_y, 0, hh - 1).long()

            indexs = vote_y * ww + vote_x
            indexs = indexs.cpu().numpy()
            counts = np.bincount(indexs, minlength=hh * ww)
            counts = counts / (np.max(counts) + 1e-10)
            counts = np.reshape(counts, (hh, ww))
            counts = torch.from_numpy(counts).float().to(device)
            vote_map[ii, 0, ...] = counts

        return vote_map


class SinglePanopticDeepLabHeadFused(nn.Module):
    def __init__(self,
                 meta: Union[IntensityMeta, AssociationMeta, DetectionMeta, ParsingMeta],
                 decoder_channels, head_channels, num_classes, class_key, dsn=False):
        super(SinglePanopticDeepLabHeadFused, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')
        self.meta = meta
        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels,
                ),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
            if class_key[i] == 'edge':
                classifier[class_key[i]].add_module(
                    "sigmoid", nn.Sigmoid()
                )

        if dsn:
            self.classifier_dsn = nn.Sequential(
                fuse_conv(decoder_channels + 32, head_channels),
                nn.Conv2d(head_channels, num_classes[0], 1)
            )
        else:
            self.classifier_dsn = None

        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def stride(self, basenet_stride):
        return basenet_stride // 4

    def forward(self, x, x_dsn=None):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred


class SinglePanopticDeepLabHead(nn.Module):
    def __init__(self,
                 meta: Union[IntensityMeta, AssociationMeta, DetectionMeta, ParsingMeta],
                 decoder_channels, head_channels, num_classes, class_key, dsn=False):
        super(SinglePanopticDeepLabHead, self).__init__()
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')
        self.meta = meta
        self.num_head = len(num_classes)
        assert self.num_head == len(class_key)

        classifier = {}
        for i in range(self.num_head):
            classifier[class_key[i]] = nn.Sequential(
                fuse_conv(
                    decoder_channels,
                    head_channels,
                ),
                nn.Conv2d(head_channels, num_classes[i], 1)
            )
            if class_key[i] == 'edge':
                classifier[class_key[i]].add_module(
                    "sigmoid", nn.Sigmoid()
                )

        if dsn:
            self.classifier_dsn = nn.Sequential(
                fuse_conv(decoder_channels + 32, head_channels),
                nn.Conv2d(head_channels, num_classes[0], 1)
            )
        else:
            self.classifier_dsn = None

        self.classifier = nn.ModuleDict(classifier)
        self.class_key = class_key

    def stride(self, basenet_stride):
        return basenet_stride // 4

    def forward(self, x, x_dsn=None):
        pred = OrderedDict()
        # build classifier
        for key in self.class_key:
            pred[key] = self.classifier[key](x)

        return pred


class SinglePanopticDeepLabDecoder(nn.Module):
    def __init__(self, in_channels, feature_key, low_level_channels,
                 low_level_key, low_level_channels_project,
                 decoder_channels, atrous_rates, aspp_channels=None, dsn=False):
        super(SinglePanopticDeepLabDecoder, self).__init__()

        if aspp_channels is None:
            aspp_channels = decoder_channels
        self.aspp = ASPP(in_channels, out_channels=aspp_channels,
                         atrous_rates=atrous_rates)

        self.feature_key = feature_key
        self.decoder_stage = len(low_level_channels)
        assert self.decoder_stage == len(low_level_key)
        assert self.decoder_stage == len(low_level_channels_project)
        self.low_level_key = low_level_key
        fuse_conv = partial(stacked_conv, kernel_size=5, num_stack=1, padding=2,
                            conv_type='depthwise_separable_conv')

        # Transform low-level feature
        project = []
        # Fuse
        fuse = []
        # Top-down direction, i.e. starting from largest stride
        for i in range(self.decoder_stage):
            project.append(
                nn.Sequential(
                    nn.Conv2d(low_level_channels[i], low_level_channels_project[i], 1, bias=False),
                    nn.BatchNorm2d(low_level_channels_project[i]),
                    nn.ReLU()
                )
            )
            if i == 0:
                fuse_in_channels = aspp_channels + low_level_channels_project[i]
            else:
                fuse_in_channels = decoder_channels + low_level_channels_project[i]
            fuse.append(
                fuse_conv(
                    fuse_in_channels,
                    decoder_channels,
                )
            )
        self.project = nn.ModuleList(project)
        self.fuse = nn.ModuleList(fuse)
        self.dsn = dsn

    def set_image_pooling(self, pool_size):
        self.aspp.set_image_pooling(pool_size)

    def forward(self, features):
        x = features[self.feature_key]

        x = self.aspp(x)

        xs = []
        # build decoder
        for i in range(self.decoder_stage):
            l = features[self.low_level_key[i]]
            l = self.project[i](l)
            x = F.interpolate(x, size=l.size()[2:], mode='bilinear', align_corners=True)
            x = torch.cat((x, l), dim=1)
            x = self.fuse[i](x)

            xs.append(x)

        if self.dsn:
            return xs[-1], xs[-2]
        return xs[-1]
