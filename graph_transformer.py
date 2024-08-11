import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_transformer_layer import GraphTransformerLayer


class GraphTransformer(nn.Module):
    def __init__(self, device, n_layers, node_dim, hidden_dim, out_dim, n_heads, dropout):
        super(GraphTransformer, self).__init__()
        
        self.device = device
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.linear_h = nn.Linear(node_dim, hidden_dim)
        
        self.layers = nn.ModuleList([GraphTransformerLayer(hidden_dim, hidden_dim, n_heads, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(n_layers - 1)])

        self.layers.append(
            GraphTransformerLayer(hidden_dim, out_dim, n_heads, dropout, self.layer_norm, self.batch_norm, self.residual))

    def forward(self, g):
        g = g.to(self.device)
        h = g.ndata['sim_feature'].float().to(self.device)
        l_layer_representations = []

        h = self.linear_h(h)

        for conv in self.layers:
            h = conv(g, h)
            l_layer_representations.append(h)

        g.ndata['h'] = h

        return l_layer_representations