#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import LSTM

D2 = 512

def conv3x3(in_channels: int, out_channels: int, padding: int = 1):
    """3x3 Convolution Layer with stride 1"""
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=3,
                     stride=1
                     padding=padding,
                     bias=True)

class FeatureExtractor(nn.Module):
    """
    Extracts feature maps from the entire formula image.
    Output shape = D2*H'*W', where D2=512 for our implementation
    """
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

class RowEncoder(nn.Module):
    """
    Encode each row of the feature extractor output.
    Output shape = H'*D2, where D2=512 for our implementattion.
    """
    def __init__(self) -> None:
        super().__init__()
        self.num_layers = 1
        self.lstm = LSTM(input_size=D2,output_size=D2,
                         num_layers=self.num_layers,
                         batch_size=True)

    def forward(self, x):
        """Expected input size = (H',W',D2) or
        (batch size, sequence length, number of features)"""
        return self.lstm(x)

class FormulaEncoder(nn.Module):
    """
    Encode the entire formula image using FeatureExtractor
    and RowEncoder.
    Input shape = H*W, Output shape = H'*D2
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.row_encoder = RowEncoder()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.row_encoder(x)
        return x
