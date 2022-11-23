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
                     stride=1,
                     padding=padding,
                     bias=True)

class FeatureExtractor(nn.Module):
    """
    Extracts feature maps from the entire formula image.
    Output shape = D2*H'*W', where D2=512 for our implementation
    """
    # NOTE: All the inputs must be of the form (1*H*W) where H and W
    # are fixed apriori (hyperparameters).
    def __init__(self) -> None:
        self.conv1 = conv3x3(in_channels=1, out_channels=64, padding=1) #64*H*W
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) #64*(H/2)*(W/2)
        self.conv2 = conv3x3(64, 128, 1) #128*(H/2)*(W/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) #128*(H/4)*(W/4)
        self.conv3 = conv3x3(128, 256, 1) #256*(H/4)*(W/4)
        self.bn3 = nn.BatchNorm2d(256) #256*(H/4)*(W/4)
        self.conv4 = conv3x3(256, 256, 1) #256*(H/4)*(W/4)
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=(1,2)) #256*(H/4)*(W/8)
        self.conv5 = conv3x3(256, 512, 1) #512*(H/4)*(W/8)
        self.bn5 = nn.BatchNorm2d(512) #512*(H/4)*(W/8)
        self.pool5 = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)) #512*(H/8)*(W/8)
        self.conv6 = conv3x3(512, 512, 0) #512*(H/8 - 2)*(W/8 - 2)
        self.bn6 = nn.BatchNorm2d(512) #512*(H/8 - 2)*(W/8 - 2) == D2*H'*W'
        super().__init__()

    def forward(self, x):
        """
        Inputs:
          - x (shape: N*1*H*W)
        Outputs:
          - y (shape: N*D2*H'*W' == N*512*(H/8 - 2)*(W/8 - 2))
        where N is the batch size.
        """
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.bn5(x)
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
                         batch_first=True)

    def forward(self, x: torch.Tensor):
        """
        Inputs:
          - x (shape: N*D2*H'*W')
        Outputs:
          - y (shape: N*H'*D2)
        where N is the batch size.
        """
        N = x.shape[0] # expecting batch size
        num_rows = x.shape[2] # expecting H' here
        w = x.shape[3] # expecting W'
        #output = torch.zeros(N*num_rows*D2, require_grad=True) # doubtful if this will work because we don't know if grads are passed to a freshly initialized tensor.
        encoding_list = []
        for r in range(num_rows):
            row = x[:,:,r,:].transpose(1,2) # shape: N*W'*D2
            _, (encoding, _) = self.lstm(row) #assuming h0 and c0 are zeros, shape: 1*N*D2
            encoding_list.append(encoding[0,:,:])
        return torch.stack(encoding_list, dim=1)

class FormulaEncoder(nn.Module):
    """
    Encode the entire formula image using FeatureExtractor
    and RowEncoder.
    Inputs:
        - x (shape: N*1*H*W)
    Outputs:
        - y (shape: N*H'*D2)
    """
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.row_encoder = RowEncoder()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.row_encoder(x)
        return x
