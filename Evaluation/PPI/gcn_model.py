# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # Define a single GCN layer using sparse matrix multiplication
# class GCNLayer(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GCNLayer, self).__init__()
#         self.weight = nn.Parameter(torch.randn(in_features, out_features))
#
#     def forward(self, adj, features):
#         # adj is expected to be a sparse tensor on the GPU
#         support = torch.mm(features, self.weight)  # Dense multiplication (X @ W)
#         output = torch.sparse.mm(adj, support)  # Sparse matrix multiplication (A_hat @ (X @ W))
#         return F.relu(output)
#
#
# # Define the complete GCN model with multiple layers
# class GCN(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
#         super(GCN, self).__init__()
#         self.layers = nn.ModuleList()
#
#         # First layer
#         self.layers.append(GCNLayer(input_dim, hidden_dim))
#
#         # Hidden layers
#         for _ in range(num_layers - 1):
#             self.layers.append(GCNLayer(hidden_dim, hidden_dim))
#
#         # Output layer (linear transformation)
#         self.out_layer = nn.Linear(hidden_dim, output_dim)
#
#     def forward(self, adj, features):
#         x = features
#         for layer in self.layers:
#             x = layer(adj, x)
#         return self.out_layer(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_sparse_adj(edge_index, num_nodes):
    """
    Create a sparse adjacency matrix (COO format) from edge indices.

    Args:
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges).
        num_nodes (int): Number of nodes in the graph.

    Returns:
        torch.sparse.Tensor: A sparse adjacency matrix in COO format.
    """
    values = torch.ones(edge_index.shape[1])  # Assign a value of 1 to each edge
    adj_matrix = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))
    return adj_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, edge_index, features, num_nodes):
        # Apply GCN operation: A_hat @ X @ W
        support = torch.mm(features, self.weight)  # X @ W

        # Create a sparse adjacency matrix from edge_index
        adj_matrix = create_sparse_adj(edge_index, num_nodes)

        # Perform sparse matrix multiplication: A_hat @ (X @ W)
        output = torch.sparse.mm(adj_matrix, support)
        return F.relu(output)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_features, hidden_features))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_features, hidden_features))
        self.out_layer = nn.Linear(hidden_features, out_features)

    def forward(self, edge_index, features):
        num_nodes = features.size(0)  # Get the number of nodes from the features tensor
        x = features
        for layer in self.layers:
            x = layer(edge_index, x, num_nodes)  # Pass num_nodes to each layer
        return self.out_layer(x)

