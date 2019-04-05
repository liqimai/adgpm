import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import normt_spm, spm_to_tensor
from models.gcn import GraphConvFunction

def FullyConnect(in_channels, out_channels, dropout=False, relu=True):
    layer = []
    if dropout:
        layer.append(nn.Dropout(p=0.5))
    layer.append(nn.Linear(in_channels, out_channels))
    if relu:
        layer.append(nn.LeakyReLU(negative_slope=0.2))
    return nn.Sequential(*layer)


class GLP(nn.Module):
    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, k):
        super().__init__()

        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj.cuda()
        self.k = k

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        layers = []
        last_c = in_channels
        for i, c in enumerate(hl):
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            layers.append(FullyConnect(last_c, c, dropout=dropout))
            last_c = c
        layers.append(FullyConnect(last_c, out_channels, relu=False, dropout=dropout_last))

        self.mlp = nn.Sequential(*layers)

    def extra_repr(self):
        return '(k): {}'.format(self.k)

    def forward(self, inputs, mask=None):
        if mask is not None:
            inputs = inputs[mask]
        inputs = self.mlp(inputs)
        return F.normalize(inputs)

    def conv(self, inputs):
        return GraphConvFunction.apply(self.adj, inputs, self.k)