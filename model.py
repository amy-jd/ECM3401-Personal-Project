import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn


class GNN(torch.nn.Module):

    def __init__ (self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, output_channels)
 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        #x = torch.log_softmax(x, dim=1 )
        return x
    
   
    

class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        self.trans = nn.TransformerEncoderLayer(d_model=embed_dim,
            nhead=num_heads,
            batch_first=True)


class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads):
        self.spatialBlock = GNN(in_channels, hidden_channels)
        self.temporalBlock = Transformer(hidden_channels, num_heads, 1)

class Model(torch.nn.Module):
    def __init__(self, num_blocks, in_channels, hidden_channels, num_heads):
        self
