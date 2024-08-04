import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# This class has been created to provide graphSage convolutional layer from the paper
# " Inductive Representation Learning on Large Graphs by Hamilton",
# It provides graphSage convolutional operation by providing graphsage mean, max and LSTM aggregations and combining by
# a non linear function.
# date created = 22 Oct, 2021, Project name : GDBMs, created by: Tariq Habib Afridi, email address: afridi@khu.ac.kr

class NeighborAggregator(nn.Module):
    """ This class provides neighbor aggregation based on three functions such as mean, sum, and max
    used in sageConv of GraphSage conv layer.
    """
    def __init__(self, input_dim, output_dim, use_bias=False, aggr_method="mean"):
        super(NeighborAggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.aggr_method = aggr_method
        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim))

        if use_bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, neighbor_feature):
        if self.aggr_method == "mean":
            neighbor_feature.mean(dim=1)
        elif self.aggr_method == "sum":
            neighbor_feature.sum(dim=1)
        elif self.aggr_method == "max":
            neighbor_feature.max(dim=1)
        else:
            raise ValueError("Unsupported aggr_method, expected mean, sum, max, but got {}".format(self.aggr_method))
        neighbor_hidden = torch.matmul(neighbor_feature, self.weight)
        if self.use_bias:
            neighbor_hidden += self.bias

        return neighbor_hidden


class sageConv(nn.Module):
    """ This class provides implementation for the GraphSage convolutional layer.
        """
    def __init__(self, input_dim, hidden_dim,
                 activation=F.relu,
                 aggr_neighbor_method="mean",
                 aggr_hid_method="sum"):

        super(sageConv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.aggr_hid_method = aggr_hid_method
        self.aggr_neighbor_method = aggr_neighbor_method

        self.weight = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))

        self.neighborAgg = NeighborAggregator(self.input_dim, self.hidden_dim, aggr_method=aggr_neighbor_method)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)

    def forward(self, src_node_features, neighbor_node_features):
        neighbor_hidden = self.neighborAgg(neighbor_node_features)
        self_hidden = torch.matmul(src_node_features, self.weight)

        if self.aggr_hid_method == "sum":
            hidden = self_hidden + neighbor_hidden
        elif self.aggr_hid_method == "concat":
            hidden = torch.cat([self_hidden, neighbor_hidden], dim=1)
        else:
            raise ValueError("Expected sum or concat, got {}".format(self.aggr_hid_method))

        if self.activation:
            hidden = self.activation(hidden)
        return hidden

#an example of using sageGCN where it created a two layer GraphSage model for node classification
class GraphSage(nn.Module, object):
    """docstring for GraphSage"""

    def __init__(self, input_dim, hidden_dim=[64, 64], num_neighbors_list=[10, 10]):
        super(GraphSage, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_neighbors_list = num_neighbors_list

        self.gcn = []
        self.gcn1 = sageConv(input_dim, hidden_dim[0])
        self.gcn2 = sageConv(hidden_dim[0], hidden_dim[1])
        self.gcn.append(self.gcn1)
        self.gcn.append(self.gcn2)

        self.num_layers = len(num_neighbors_list)

    def forward(self, node_feature_list):
        hidden = node_feature_list
        for l in range(self.num_layers):
            next_hidden = []
            gcn = self.gcn[l]
            for hop in range(self.num_layers - l):
                src_nodes = hidden[hop]
                src_nums = len(src_nodes)
                h = gcn(src_nodes, hidden[hop + 1].view(src_nums, self.num_neighbors_list[l], -1))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]