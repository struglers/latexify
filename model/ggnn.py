#!/usr/bin/env python3

from torch_geometric.nn import GatedGraphConv

D2 = 512
K = 2
def GatedGNN():
    return GatedGraphConv(out_channels=D2,
                          num_layers=K,
                          aggr="add",
                          bias=True)

def main():
    model = GatedGNN()

if __name__ == '__main__':
    main()
