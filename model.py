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
        return x
   
class Transformer(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers):
        super().__init__()
        self.trans = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
    
    def forward(self, x):
        x = self.trans(x)
        return x 

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads):
        super().__init__()
        self.spatialBlock = GNN(input_channels, hidden_channels, output_channels)
        self.temporalBlock = Transformer(hidden_channels, num_heads, 1)

    def forward(self, x, edge_index):
        x = self.spatialBlock(x, edge_index)
        x = self.reshape_to_trans(x)
        x = self.temporalBlock(x)
        x = self.reshape_to_graph(x)
        return x
    
    def reshape_to_trans(self, x):
        x = x.unsqueeze(-1)
        x = x.repeat(1, 1, self.embed_dim)
        return x
    
    def reshape_to_graph(self,x):
        return x

class Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.GNN = GNN(input_channels, hidden_channels, output_channels)
        self.prediction = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.GNN(x, edge_index)
        x = self.prediction(x)
        return x
