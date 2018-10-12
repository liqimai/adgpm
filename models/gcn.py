import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.init import xavier_uniform_

from utils import normt_spm, spm_to_tensor

class GraphConvFunction(Function):

    @staticmethod
    def forward(ctx, adj, x, k):
        ctx.conv_k = k
        ctx.save_for_backward(adj)
        for _ in range(k):
            x = torch.mm(adj, x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        adj, = ctx.saved_tensors
        k = ctx.conv_k
        assert ctx.needs_input_grad[0] == False, "Gradients of adj is not supported."
        assert ctx.needs_input_grad[2] == False, "Gradients of k is not supported."

        grad_x = None
        if ctx.needs_input_grad[1]:
            adj_t = adj.t()
            grad_x = grad_output
            for _ in range(k):
                grad_x = torch.mm(adj_t, grad_x)

        return None, grad_x, None


class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True, k=1):
        super().__init__()

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        self.k = k
        xavier_uniform_(self.w)

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        inputs = torch.mm(inputs, self.w)
        inputs = GraphConvFunction.apply(adj, inputs, self.k)
        outputs = inputs + self.b

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return '(k): {}'.format(self.k)

class GCN(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, ks):
        super().__init__()

        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                            shape=(n, n), dtype='float32')
        adj = normt_spm(adj, method='in')
        adj = spm_to_tensor(adj)
        self.adj = adj.cuda()

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False

        i = 0
        layers = []
        last_c = in_channels
        for c, k in zip(hl, ks):
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout, k=k)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last, k=ks[-1])
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x):
        for conv in self.layers:
            x = conv(x, self.adj)
        return F.normalize(x)

