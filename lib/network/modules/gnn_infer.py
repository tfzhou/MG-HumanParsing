import functools

import torch
import torch.nn as nn
from inplace_abn.bn import InPlaceABNSync
from lib.modules import ConvGRU


BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
from lib.modules.dcn import DFConv2d
class Decomposition(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Decomposition, self).__init__()
        self.att_fh = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2 * hidden_dim), nn.LeakyReLU(inplace=False))
        self.att_fh1=nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, xf, xh):
        att_fh = self.att_fh(torch.cat([xf, xh], dim=1))
        att = self.att_fh1(att_fh)
        return att

class conv_Update(nn.Module):
    def __init__(self, hidden_dim=10, paths_len=3):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        # detect if CUDA is available or not
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            dtype = torch.cuda.FloatTensor  # computation in GPU
        else:
            dtype = torch.FloatTensor
        self.conv_update = ConvGRU(input_dim=paths_len* hidden_dim,
                        hidden_dim=hidden_dim,
                        kernel_size=(3,3),
                        num_layers=1,
                        dtype=dtype,
                        batch_first=True,
                        bias=True,
                        return_all_layers=False)

    def forward(self, x, message_list):
        if len(message_list)>1:
            _, out = self.conv_update(torch.cat(message_list, dim=1).unsqueeze(1), [x])
        else:
            _, out = self.conv_update(message_list[0].unsqueeze(1), [x])
        return out[0][0]

# class conv_Update(nn.Module):
#     def __init__(self, hidden_dim=10, paths_len=3):
#         super(conv_Update, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.conv_update = nn.Sequential(
#             nn.Conv2d((paths_len+1) * hidden_dim, 2 * hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
#             BatchNorm2d(2 * hidden_dim), nn.LeakyReLU(inplace=False),
#             nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
#             BatchNorm2d(hidden_dim), nn.LeakyReLU(inplace=False)
#         )
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.relu = nn.LeakyReLU()
#
#     def forward(self, x, message_list):
#         if len(message_list)>1:
#             out = self.conv_update(torch.cat([x]+message_list, dim=1))
#         else:
#             out = self.conv_update(torch.cat([x, message_list[0]], dim=1))
#         return self.relu(self.gamma*x+out)

class Composition(nn.Module):
    def __init__(self, hidden_dim, parts_len):
        super(Composition, self).__init__()
        self.conv_ch = nn.Sequential(
            nn.Conv2d((parts_len) * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.LeakyReLU(inplace=False),
            nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.LeakyReLU(inplace=False)
        )
    def forward(self, xp_list):
        xph = self.conv_ch(torch.cat(xp_list, dim=1))
        return xph
class Part_Dependency(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Part_Dependency, self).__init__()
        self.dconv = nn.Sequential(
            DFConv2d(
                2 * hidden_dim,
                2 * hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
            DFConv2d(
                2 * hidden_dim,
                hidden_dim,
                with_modulated_dcn=True,
                kernel_size=3,
                stride=1,
                groups=1,
                dilation=1,
                deformable_groups=1,
                bias=False
            ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, pA, pB):
        A_diffuse = self.dconv(torch.cat([pA, pB], dim=1))
        return A_diffuse


# class Part_Dependency(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
#         super(Part_Dependency, self).__init__()
#         self.cls_p = cls_p
#
#         self.dconv = nn.Sequential(
#             DFConv2d(
#                 2 * hidden_dim,
#                 2 * hidden_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
#             DFConv2d(
#                 2 * hidden_dim,
#                 hidden_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
#         )
#
#     def forward(self, pA, pB, A_att, B_att):
#         A_diffuse = self.dconv(torch.cat([pB, pA], dim=1))
#         A_diffuse_att = (1 - A_att) * A_diffuse
#         return A_diffuse_att
#
# class Part_Dependency(nn.Module):
#     def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
#         super(Part_Dependency, self).__init__()
#         self.cls_p = cls_p
#
#         self.dconv = nn.Sequential(
#             DFConv2d(
#                 2 * hidden_dim,
#                 2 * hidden_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(2 * hidden_dim), nn.ReLU(inplace=False),
#             DFConv2d(
#                 2 * hidden_dim,
#                 hidden_dim,
#                 with_modulated_dcn=True,
#                 kernel_size=3,
#                 stride=1,
#                 groups=1,
#                 dilation=1,
#                 deformable_groups=1,
#                 bias=False
#             ), BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
#         )
#
#     def forward(self, pA, pB, A_att, B_att):
#         A_diffuse = self.dconv(torch.cat([pB, pA], dim=1))
#         A_diffuse_att = (1 - A_att) * A_diffuse
#         A2B = A_diffuse_att * B_att
#         return A2B