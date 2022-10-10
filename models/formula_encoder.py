#!/usr/bin/env python3

import torch
from torch import nn

def conv3x3(in_channels: int, out_channels: int, padding: int = 1):
    """3x3 Convolution Layer with stride 1"""
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=1
                     padding=padding,
                     bias=True)

class FormulaDecoder(nn.Module):
    """Decoder for the entire formula image.
    Output shape = L*D2, where D2=512 for our implementation"""
    def __init__(self) -> None:
        self.conv1 = conv3x3(in_channels=1, out_channels=64, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = conv3x3(64, 128, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv3x3(128, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = conv3x3(256, 256, 1)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2))
        self.conv5 = conv3x3(256, 512, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1))
        self.conv6 = conv3x3(512, 512, 0)
        self.bn6 = nn.BatchNorm2d(512)
        super().__init__()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        return x
