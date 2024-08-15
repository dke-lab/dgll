import torch.nn as nn
from dgl.nn import GraphConv
from dgl.nn import SAGEConv
import torch.nn.functional as F
# GCN Model Definition
class GCN_Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN_Model, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

# GraphSAGE Model
class GraphSAGE_Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GraphSAGE_Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='gcn')
        self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='gcn')
        self.h_feats = h_feats

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h

# Define your custom GNN model here
class Custom_GNN_Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Custom_GNN_Model, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, mfgs, x):
        h_dst = x[:mfgs[0].num_dst_nodes()]
        h = self.conv1(mfgs[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:mfgs[1].num_dst_nodes()]
        h = self.conv2(mfgs[1], (h, h_dst))
        return h