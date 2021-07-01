import torch
import torch.nn as nn
import torch.nn.functional as F
import encoding
from inplace_abn.bn import InPlaceABNSync

# encoding.nn.BatchNorm1d(ncodes),
class EncModule(nn.Module):
    def __init__(self, in_channels, nclass, ncodes=32, se_loss=True):
        super(EncModule, self).__init__()
        self.se_loss = se_loss
        self.encoding = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            InPlaceABNSync(in_channels),
            encoding.nn.Encoding(D=in_channels, K=ncodes),
            # encoding.nn.BatchNorm1d(ncodes),
            nn.BatchNorm1d(ncodes),
            nn.ReLU(inplace=False),
            encoding.nn.Mean(dim=1))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid())
        if self.se_loss:
            self.selayer = nn.Linear(in_channels, nclass)

    def forward(self, x):
        en = self.encoding(x)
        b, c, _, _ = x.size()
        gamma = self.fc(en)
        y = gamma.view(b, c, 1, 1)
        outputs = [F.relu_(x + x * y)]
        if self.se_loss:
            outputs.append(self.selayer(en))
        return tuple(outputs)
