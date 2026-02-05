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
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]
            edge_index (Tensor): The relationships between nodes

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        return x

class PositionalEncoding(torch.nn.Module):
    """
    Implementing sinusoidal positional encoding, based on the approach used in 'Attention is all you need'

    Code for this class has been taken from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
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
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]

        Returns
            Tensor: The network nodes and featureswith added positional encoding, with shape [num_nodes, num_timesteps]
        """
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x

   
class TemporalTransformer(torch.nn.Module):
    """
    Transformer component which takes a timeseries sequence, and forecasts future values for it

    Parts of the component:
        -   Preprocessing: reformating the input from shape [num_nodes, num_timesteps] to [num_nodes, num_timesteps, embed_dim]
        -   Positional encoding: Adding information such that the model knows the order of the time series
        -   Computing attention mask: ensuring the model knows which timesteps can pay attention to which other ones
        -   Transformer layer: Computes predictions for the masked nodes
        -   Postprocessing: reformatting the output back to shape [num_nodes, num_timesteps] by reducing the feature dimension back to 1
    """
    def __init__(self, num_heads, embed_dim, device = 'cpu'):
        """
        Parameters:
            num_heads (int): The number of attention head the transformer encoder uses
            embed_dim (int): Dimensionality of the embedding space representing each timestep of the input data
            device (str): The device that the model is run on
        """

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
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        # Adding an extra dimension so it is (batch_size, seq_len, 1)
        x = x.unsqueeze(-1)
        # Increasing the feature dimensions from 1 to embed_dim
        x = self.input_embedding(x) 
        # Adding positional encoding so the temporal order is known by the model
        x = self.positional_encoding(x) 

        # Creating an attention mask, so that the model cannot see future time steps
        attention_mask = self.generate_attention_mask() 
        # Passing through a transformer layer
        x = self.trans(x, src_mask=attention_mask)

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
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads, embed_dim):
        super().__init__()
        self.spatialBlock = GNN(input_channels, hidden_channels, output_channels)
        self.temporalBlock = TemporalTransformer(num_heads, embed_dim)

    def forward(self, x, edge_index):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]
            edge_index (Tensor): The relationships between nodes

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        x = self.spatialBlock(x, edge_index)
        x = self.temporalBlock(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads, embed_dim):
        super().__init__()
        self.spatio_temporal = SpatioTemporalBlock(input_channels, hidden_channels, output_channels, num_heads, embed_dim)
        self.prediction = nn.Linear(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]
            edge_index (Tensor): The relationships between nodes

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        x = self.spatio_temporal(x, edge_index)
        x = self.prediction(x)
        return x
