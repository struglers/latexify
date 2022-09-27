#!/usr/bin/env python3

"""Implementing the encoder required for feature extraction of symbols"""

import torch
from torch import nn

# TODO: Implement LResNet50E-IR

def conv3x3(in_channels: int, out_channels: int, stride: int = 1, groups: int = 1, dilation: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=num_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channels=num_channels,
                               out_channels=num_channels,
                               kernel_size=3, stride=2, padding=1)

        self.bn3 = nn.BatchNorm2d(num_features=num_channels)
        self.conv3 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=num_channels)
        self.prelu2 = nn.PReLU()
        self.conv4 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels,
                               kernel_size=3, stride=1, padding=1)

        self.conv_res = nn.Conv2d(num_channels, num_channels, kernel_size=1,
                                  stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(num_features=num_channels)

    def forward(self, x):
        pass


class LResNet50EIR(nn.Module):
    """Used for extracting features of symbol region"""
    def __init__(self):
        super(LResNet50EIR, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU()
