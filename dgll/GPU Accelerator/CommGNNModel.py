import torch as th
import torch.nn as nn
import torch.functional as F
import dgll
import dgll.nn as dgllnn
import sklearn.linear_model as lm
import sklearn.metrics as skm
import tqdm

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()

        assert n_layers > 1
        # input layer
        self.layers.append(dgllnn.GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layer
        for i in range(1, n_layers - 1):
            self.layers.append(dgllnn.GraphConv(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(dgllnn.GraphConv(n_hidden, n_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        if blocks[0].is_block:
            # normal gcn
            h = x
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.dropout(h)
        else:
            # transductive mos
            subg, block = blocks
            h = x
            for layer in self.layers[:-1]:
                # print(50*"++")
                # print(h.shape)
                # print(subg)
                # print(50*"++")
                h = layer(subg, h)

                h = self.dropout(h)

            # slice out the input nodes for block
            internal_input_nids = block.ndata[dgll.NID]['_N'].to('cuda')
            h = self.layers[-1](block, h[internal_input_nids])
        return h




class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dgllnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dgllnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dgllnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dgllnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        if isinstance(blocks, dgll.dgllGraph):
            # mos inductive
            assert isinstance(blocks, dgll.dgllGraph)
            h = x
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
        else:
            assert isinstance(blocks, list)
            if blocks[0].is_block:
                # normal sage
                h = x
                for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                    h = layer(block, h)
                    if l != len(self.layers) - 1:
                        h = self.activation(h)
                        h = self.dropout(h)
            else:
                # mos transductive
                subg, block = blocks
                h = x
                for layer in self.layers[:-1]:
                    h = layer(subg, h)
                    h = self.activation(h)
                    h = self.dropout(h)

                # slice out the input nodes for block
                internal_input_nids = block.ndata[dgll.NID]['_N'].to('cuda')
                h = self.layers[-1](block, h[internal_input_nids])

        return h
