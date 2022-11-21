#!/usr/bin/env python3

import torch
from torch import nn

class CoordinateEncoder(nn.Module):
    """The coordinate encoder first takes input of dimension (N*4*L)
    where N is the batch size, 4 is the number of channels and L is the number
    of symbols in the formula image. num_symbols is the sequence length.
    3x3 convolutions with padding 1 is used to preserve the sequence length.

    The idea is to combine similar coordinates of all the symbols to generate
    a 'global' representation of all the coordinates. The output is of shape
    (N*128*L).

    The 1x1 convolution output is transposed to shape (N*L*128), so
    that now each 128 dimensional vector is considered to be a single feature
    vector for each symbol. This set of transposed features is fed into a
    bidirectional LSTM to get outputs of size N*L*256"""
    def __init__(self) -> None:
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 128, 3, stride=1, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=128,
                            num_layers=1, bias=True, batch_first=True,
                            dropout=0, bidirectional=True)
        super().__init__()

    def __call__(self, x: torch.Tensor):
        """
        Input shape: (N*4*L)
        Output shape: (N*L*256)
        """
        x = self.conv1(x) #shape: N*32*L
        x = self.conv2(x) #shape: N*128*L
        x = x.transpose(1,2) #Change dimension to N*L*128
        x, _ = self.lstm(x) #outputs from both directions are concatenated
        return x #Has dimension N*L*256.
