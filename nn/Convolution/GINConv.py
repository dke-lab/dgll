# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 04:41:24 2024

@author: Tariq
"""

import torch

class GinConv(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, Adj, Feat):
        """
        Params
        ------
        Adj [batch x nodes x nodes]: adjacency matrix
        Feat [batch x nodes x features]: node features matrix

        Returns
        -------
        X' [batch x nodes x features]: updated node features matrix
        """
        X = self.linear(Feat + Adj @ Feat)
        X = torch.nn.functional.relu(X)

        return X


#an example of using ginConv where it created a multi layer GIN model.
class GIN(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super().__init__()

        self.in_proj = torch.nn.Linear(input_dim, hidden_dim)

        self.convs = torch.nn.ModuleList()

        for _ in range(n_layers):
            self.convs.append(GinConv(hidden_dim))

        # In order to perform graph classification, each hidden state
        # [batch x nodes x hidden_dim] is concatenated, resulting in
        # [batch x nodes x hiddem_dim*(1+n_layers)], then aggregated
        # along nodes dimension, without keeping that dimension:
        # [batch x hiddem_dim*(1+n_layers)].
        self.out_proj = torch.nn.Linear(hidden_dim * (1 + n_layers), output_dim)

    def forward(self, A, X):
        X = self.in_proj(X)

        hidden_states = [X]

        for layer in self.convs:
            X = layer(A, X)
            hidden_states.append(X)

        X = torch.cat(hidden_states, dim=2).sum(dim=1)

        X = self.out_proj(X)

        return X