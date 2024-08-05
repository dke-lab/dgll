import numpy as np
from dgll import backend as F


### This class has been created to provide graph attention network (GAT) layer from the paper
# "Graph Attention Networks by Petar Veličković",
# It provides GAT operation by first computing attention scores and then aggregating and combining by Relu.
# date created = 27 Oct, 2021, Project name : GDBMs, created by: Tariq Habib Afridi, email address: afridi@khu.ac.kr

class gatConv(F.nn.Module):
    """
    Simple GAT layer Implementation for paper "Graph Attention Networks by Petar Veličković".
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(gatConv, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = F.Parameter(F.empty(size=(in_features, out_features)))
        F.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = F.Parameter(F.empty(size=(2 * out_features, 1)))
        F.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = F.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = F.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        e = self._prepare_attentional_mechanism_input(Wh)  # this prepares the attention mechanism input for GAT

        zero_vec = -9e15 * F.ones_like(e)
        attention = F.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = F.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = F.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = F.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpecialSpmmFunction(F.autograd.Function):
    """Special function created for sparse region backpropataion layer in GAT."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = F.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return F.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(F.nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class sparseGatConv(F.nn.Module):
    """
    Sparse version GAT layer Implementation for paper "Graph Attention Networks by Petar Veličković".
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(sparseGatConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = F.Parameter(F.zeros(size=(in_features, out_features)))
        F.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = F.Parameter(F.zeros(size=(1, 2 * out_features)))
        F.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = F.Dropout(dropout)
        self.leakyrelu = F.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()  # utilizes the special

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = F.mm(input, self.W)
        # h: N x out
        assert not F.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = F.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = F.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not F.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, F.Size([N, N]), F.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, F.Size([N, N]), h)
        assert not F.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not F.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(F.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [gatConv(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = gatConv(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class SpGAT(F.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [sparseGatConv(nfeat,
                                         nhid,
                                         dropout=dropout,
                                         alpha=alpha,
                                         concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = sparseGatConv(nhid * nheads,
                                     nclass,
                                     dropout=dropout,
                                     alpha=alpha,
                                     concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)