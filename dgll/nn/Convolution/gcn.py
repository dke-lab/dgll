# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 04:44:05 2024

@author: Tariq
"""

import numpy as np
from dgll import backend as F
import math
r"""
This class has been developed to accomodate various spatial convolution graph algorithms
Initial Graph convolution network, Graph Attention network and GraphSage has been implemented
User can create multiple layer GCN. GraphSage and GAT models using methods from these class
"""

class GraphConvolution(F.nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = F.Parameter(F.FloatTensor(in_features, out_features))
        if bias:
            self.bias = F.Parameter(F.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        support = F.mm(x, self.weight) # matrix multiplication
        output = F.spmm(adj, support)  # sparse matrix multiplication
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(F.nn.Module):
    def __init__(self, in_features, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.gcn1 = GraphConvolution(in_features, nhid)
        self.gcn2 = GraphConvolution(nhid, nclass)

    def forward(self, x, adj):
        h1 = F.relu(self.gcn1(x, adj))
        h1_d = F.dropout(h1, self.dropout, training=self.training)
        logits = self.gcn2(h1_d, adj)
        output = F.log_softmax(logits, dim=1)
        return output