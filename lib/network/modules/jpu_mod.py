import torch
import torch.nn as nn

from torch.nn import functional as F


class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class simple_JPU(nn.Module):
    def __init__(self, in_channels, width=256, norm_layer=None):
        super(simple_JPU, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=False))

        self.dilation1 = nn.Sequential(SeparableConv2d(in_channels, width, kernel_size=3, padding=1, dilation=1, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation2 = nn.Sequential(SeparableConv2d(in_channels, width, kernel_size=3, padding=2, dilation=2, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation3 = nn.Sequential(SeparableConv2d(in_channels, width, kernel_size=3, padding=4, dilation=4, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation4 = nn.Sequential(SeparableConv2d(in_channels, width, kernel_size=3, padding=8, dilation=8, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(
            nn.Conv2d(width*4, in_channels, 1, padding=0, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=False))

    def forward(self, c2, c4):
        _, _, h, w = c2.size()
        c4 = F.interpolate(c4, (h, w), mode='nearest', align_corners=None)
        feat = self.conv1(c2) + c4
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)
        feat = self.conv2(feat)
        return feat

class JPU(nn.Module):
    def __init__(self, in_channels, width=512, norm_layer=None):
        super(JPU, self).__init__()

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels[-1], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=False))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels[-2], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=False))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels[-3], width, 3, padding=1, bias=False),
            norm_layer(width),
            nn.ReLU(inplace=False))

        self.dilation1 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=1, dilation=1, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation2 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=2, dilation=2, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation3 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=4, dilation=4, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation4 = nn.Sequential(SeparableConv2d(3*width, width, kernel_size=3, padding=8, dilation=8, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))

    def forward(self, *inputs):
        feats = [self.conv5(inputs[-1]), self.conv4(inputs[-2]), self.conv3(inputs[-3])]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='nearest', align_corners=None)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='nearest', align_corners=None)
        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return inputs[0], inputs[1], inputs[2], feat


class hr_JPU(nn.Module):
    def __init__(self, in_channels, width=128, norm_layer=None):
        super(hr_JPU, self).__init__()

        self.channels =sum(in_channels)
        self.dilation1 = nn.Sequential(SeparableConv2d(self.channels, width, kernel_size=3, padding=1, dilation=1, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation2 = nn.Sequential(SeparableConv2d(self.channels, width, kernel_size=3, padding=2, dilation=2, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation3 = nn.Sequential(SeparableConv2d(self.channels, width, kernel_size=3, padding=4, dilation=4, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))
        self.dilation4 = nn.Sequential(SeparableConv2d(self.channels, width, kernel_size=3, padding=8, dilation=8, bias=False, norm_layer=norm_layer),
                                       norm_layer(width),
                                       nn.ReLU(inplace=False))

    def forward(self, inputs):
        feats = [inputs[-1], inputs[-2], inputs[-3], inputs[-4]]
        _, _, h, w = feats[-1].size()
        feats[-2] = F.interpolate(feats[-2], (h, w), mode='nearest', align_corners=None)
        feats[-3] = F.interpolate(feats[-3], (h, w), mode='nearest', align_corners=None)
        feats[-4] = F.interpolate(feats[-4], (h, w), mode='nearest', align_corners=None)

        feat = torch.cat(feats, dim=1)
        feat = torch.cat([self.dilation1(feat), self.dilation2(feat), self.dilation3(feat), self.dilation4(feat)], dim=1)

        return feat