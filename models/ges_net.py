import torch
from mypath import Path
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

class Extractor(nn.Module):
    def __init__(self,lmd1,lmd2):
        super(Extractor, self).__init__()
        inplane = 1
        outplane = 32
        midplanes = int(32 * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer1 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max1 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=0, stride=2))

        inplane = 32
        outplane = 128
        midplanes = int(inplane * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer2 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max2 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1, stride=2))

        inplane = 128
        outplane = 256
        midplanes = int(inplane * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer3 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max3 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1, stride=2))

        inplane = 256
        outplane = 256
        midplanes = int(inplane * lmd2)
        mid = int(4 * outplane * lmd1)
        self.layer4 = nn.Sequential(
            nn.Conv3d(inplane, mid, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, midplanes, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1), bias=False),
            nn.BatchNorm3d(midplanes, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, outplane, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(outplane, eps=0.001, momentum=0.9, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))
        self.max4 = nn.Sequential(nn.MaxPool3d(kernel_size=(2, 2, 2), padding=1, stride=2))
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.5)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(256 * 3 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.layer1(x)
        # x = self.dropout(x)
        x = self.max1(x)

        x = self.layer2(x)
        # x = self.dropout(x)
        x = self.max2(x)

        x = self.layer3(x)
        x = self.dropout1(x)
        x = self.max3(x)

        x = self.layer4(x)
        x = self.dropout1(x)
        x = self.max4(x)
        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x