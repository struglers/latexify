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

def conv1x1(in_channels: int, out_channels: int, stride: int = 1):
    """1x1 convolution with no padding"""
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False)

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = conv3x3(in_channels=16, out_channels=num_channels)
        self.bn2 = nn.BatchNorm2d(num_features=num_channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = conv3x3(in_channels=num_channels, out_channels=num_channels, stride=2)

        self.bn3 = nn.BatchNorm2d(num_features=num_channels)
        self.conv3 = conv3x3(in_channels=num_channels, out_channels=num_channels)
        self.bn4 = nn.BatchNorm2d(num_features=num_channels)
        self.prelu2 = nn.PReLU()
        self.conv4 = conv3x3(in_channels=num_channels, out_channels=num_channels)

        self.conv_res = conv1x1(num_channels, num_channels, stride=1)
        self.bn5 = nn.BatchNorm2d(num_features=num_channels)

    def forward(self, x):
        pass


class LResNet50EIR(nn.Module):
    """Used for extracting features of symbol region"""
    def __init__(self):
        super(LResNet50EIR, self).__init__()
        self.conv1 = conv3x3(in_channels=16, out_channels=32)
        self.bn1 = nn.BatchNorm2d(16)
        self.prelu = nn.PReLU()
