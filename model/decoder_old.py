#!/usr/bin/env python3

import torch
from torch import nn
from torch.nn import LSTM

D2 = 512

class AttentionLayer(nn.Module):
    """Generates context vectors"""
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(D2, D2, bias=False)
        self.w2 = nn.Linear(D2, D2, bias=False)
        self.beta = nn.Linear(D2, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h_t, v):
        """
        Compute context vectors:
        Input size = h_t = (1, D2), v = (v_len, D2)
        Output size = ()
        """
        a_t = self.beta(self.tanh(self.w1(h_t) + self.w2(v))) # dim = (v_len, 1)
        a_t = self.softmax(a_t) # dim = (v_len, 1)
        return a_t.transpose(0,1) @ v # context vector c_t, dim = (1, D2)

class OutputDecoder(nn.Module):
    """
    Gives the output sequence
    Refer to the following link to set up minibatches for variable length sequences:
    https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e

    The following link says that the usage of LSTM over LSTMCell is preferable:
    https://discuss.pytorch.org/t/using-packedsequence-with-lstmcell/2003/4
    """
    def __init__(self, embedding_size=80, num_tokens=583) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_embeddings=self.num_tokens,
                                      embedding_dim=self.embedding_size)
        self.lstm = LSTM(input_size=(self.embedding_size+D2),
                         hidden_size=D2,
                         num_layers=2)
        self.attention = AttentionLayer()
        self.w_c = nn.Linear(2*D2,D2)
        self.tanh = nn.Tanh()
        self.w_out = nn.Linear(D2,self.num_tokens)
        self.softmax = nn.Softmax()
        self.start_token = 0
        self.end_token = 1

    def forward(self, h_0, w, v):
        """
        Input shape:
            h_0: (H', D2)
            w: ()
            v: (L+H', D2)
        """
        h_0 = h_0.mean(dim=0)
        c_0 = self.attention(h_0, v)
        o_0 = self.get_o(h_0, c_0)

        h_t, (hs_t, ss_t) = self.lstm(torch.concat([w, o_0]), (h_0, h_0))

    def get_o(self, h_t, c_t):
        """
        Input shape:
            h_t: (1,D2)
            c_t: (1,D2)
        """
        return self.tanh(self.w_c(torch.concat([h_t, c_t])))
