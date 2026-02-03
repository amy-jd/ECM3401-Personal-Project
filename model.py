import torch
import math
import hp
from torch_geometric.nn import GCNConv
from torch import nn, Tensor

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

# Positional Encoding - https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x

   
class TemporalTransformer(torch.nn.Module):
    def __init__(self, num_heads, num_layers, embed_dim = 16, device = 'cpu'):
        super().__init__()
        # Class parameters
        self.device = device
        self.context_window = hp.CONTEXT_WINDOW
        self.forecast_window = hp.FORECAST_WINDOW
        self.total_window = self.context_window + self.forecast_window

        # Preprocessing layers
        self.input_embedding  = nn.Linear(1, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        # Tranformer layers
        self.trans = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        # FCN layers
        self.fcn = nn.Linear(embed_dim, 1)
    
    def forward(self, x):
        # Adding an extra dimension so it is (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        # Increasing the feature dimensions from 1 to embed_dim
        x = self.input_embedding(x) 
        # Adding positional encoding so the temporal order is known by the model
        x = self.positional_encoding(x) 

        # Creating an attention mask, so that the model cannot see future time steps
        attention_mask = self.generate_attention_mask() 
        # Passing through a transformer layer
        x = self.trans(x, mask=attention_mask)

        # Reducing the feature dimensions back to 1
        x = self.fcn(x)
        # Removing the last dimensions to return it to (batch_size, seq_len)
        x = x.squeeze(-1)
        return x 
    
    def generate_attention_mask(self):
        mask = torch.ones(self.total_window, self.total_window, dtype=torch.bool)
        mask[:self.context_window, :self.context_window] = False
        mask[self.context_window:, :self.context_window] = False
        return mask

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads):
        super().__init__()
        self.spatialBlock = GNN(input_channels, hidden_channels, output_channels)
        self.temporalBlock = TemporalTransformer(hidden_channels, num_heads, 1)

    def forward(self, x, edge_index):
        x = self.spatialBlock(x, edge_index)
        x = self.temporalBlock(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.spatio_temporal = SpatioTemporalBlock(input_channels, hidden_channels, output_channels)
        self.prediction = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = self.spatio_temporal(x, edge_index)
        x = self.prediction(x)
        return x
