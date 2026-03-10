import torch
import math
import hyperparameters as hp
from torch_geometric.nn import GCNConv
from torch import nn, Tensor

class GNN(torch.nn.Module):
    def __init__ (self, input_channels, hidden_channels, output_channels):
        super().__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, output_channels)
 
    def forward(self, x, edge_index):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]
            edge_index (Tensor): The relationships between nodes

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        x = self.conv1(x, edge_index)
        x = torch.relu(x) #
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
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
            Tensor: The network nodes and features with added positional encoding, with shape [num_nodes, num_timesteps]
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
    def __init__(self, num_heads, embed_dim, context_window, forecast_window, context_dim, device = 'cpu'):
        """
        Parameters:
            num_heads (int): The number of attention head the transformer encoder uses
            embed_dim (int): Dimensionality of the embedding space representing each timestep of the input data
            device (str): The device that the model is run on
        """

        super().__init__()
        # Class parameters
        self.device = device

        #self.context_to_gate = nn.Linear(context_dim, embed_dim)


        # Tranformer layers
        self.trans1 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.trans2 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.trans3 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.trans4 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.trans5 = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

    
    def forward(self, x, attention_mask, context):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """

        # Passing through a transformer layer
        x = self.trans1(x, src_mask=attention_mask)
        x = self.trans2(x, src_mask=attention_mask)
        x = self.trans3(x, src_mask=attention_mask)
        #x = self.trans4(x, src_mask=attention_mask)
        #x = self.trans5(x, src_mask=attention_mask)

        return x 
    
    def generate_attention_mask(self):
        mask = torch.ones(self.total_window, self.total_window, dtype=torch.bool)
        mask[:self.context_window, :self.context_window] = False
        mask[self.context_window:, :self.context_window] = False
        return mask


class TemporalContextEmbedding(torch.nn.Module):
    def __init__(self, context_dim, embed_dim):
        super().__init__()
        self.time_embedding = nn.Embedding(6,8) # remove hard coding at some point
        self.week_Embedding = nn.Embedding(2,4)
        self.season_embedding = nn.Embedding(4,6)

        self.temporal_proj = nn.Linear(8 + 4 + 6, embed_dim)

    def forward(self, context):
        time_of_day = self.time_embedding(context[0])
        day_of_week = self.week_Embedding(context[1])
        season = self.season_embedding(context[2])

        all_context = torch.cat((time_of_day, day_of_week, season), dim=-1)

        context_embedded = self.temporal_proj(all_context)

        # Adding an extra dimension so it is (1, seq_len, embed_dim) and can be added to the input embedding
        context_embedded = context_embedded.unsqueeze(0)

        return context_embedded

class SpatioTemporalBlock(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads, embed_dim, context_window, forecast_window, context_dim):
        super().__init__()
        self.context_window = context_window
        self.forecast_window = forecast_window
        self.total_window = self.context_window + self.forecast_window

        self.temporal_context_embedding = TemporalContextEmbedding(context_dim, embed_dim)
        
        self.spatialBlock = GNN(input_channels, hidden_channels, output_channels)

        # Preprocessing layers
        self.input_embedding  = nn.Linear(1, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)

        self.temporalBlock = TemporalTransformer(num_heads, embed_dim, context_window, forecast_window, context_dim)

        # FCN layers
        self.fcn = nn.Linear(embed_dim, 1)

    def forward(self, x_masked, edge_index, temporal_context, weather_context, prediction_mask, masked_only=False):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]
            edge_index (Tensor): The relationships between nodes
            prediction_mask (Tensor): shape [num_nodes, num_timesteps] 

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        # GNN runs on ALL nodes so spatial context propagates to masked nodes
        x_masked = self.spatialBlock(x_masked, edge_index)

        # Only run the expensive temporal layers on masked nodes
        if masked_only:
            node_mask = prediction_mask.all(dim=1)
            x_masked = x_masked[node_mask]

        # Adding an extra dimension so it is (num_nodes, seq_len, 1)
        x_masked = x_masked.unsqueeze(-1)
        # Increasing the feature dimensions from 1 to embed_dim
        x_masked = self.input_embedding(x_masked) 
        # Adding positional encoding so the temporal order is known by the model
        x_masked = self.positional_encoding(x_masked) 

        temporal_context_embedding = self.temporal_context_embedding(temporal_context)

        # Adding in the temporal context
        x_masked = x_masked + temporal_context_embedding

        # Creating an attention mask, so that the model cannot see future time steps
        attention_mask = self.generate_attention_mask() 

        x_masked = self.temporalBlock(x_masked, attention_mask, temporal_context)

        # Reducing the feature dimensions back to 1
        x_masked = self.fcn(x_masked)
        # Removing the last dimension to return it to (num_nodes, seq_len)
        x_masked = x_masked.squeeze(-1)

        # Place transformed masked-node outputs back into a full-sized tensor

        if masked_only:
            x_out = torch.zeros_like(x_masked)
            x_out[node_mask] = x_masked
            return x_out

        return x_masked
    
    def generate_attention_mask(self):
        """
        Generates the mask for the transformer component, which determines which time steps of the input data can pay attention to which other time steps.
        Each row corresponds to a time step, and the columns correspond to which time steps it can pay attention to.

        How it works:
        - All items in the context window can pay attention to eachother
        - Items in the context window cannot pay attention to any items in the forecast window
        - Items in the forecast window can pay attention to items in the context window
        - Items in the forecast window cannot pay attention to each other, as this would be giving them access to future information

        Returns:
            Tensor: The attention mask with shape [total_window, total_window]
        """
        # Creating a mask where False means pay attention, and True means do not pay attention
        mask = torch.ones(self.total_window, self.total_window, dtype=torch.bool)

        # Setting all steps in the context window to be able to pay attention to each other
        mask[:self.context_window, :self.context_window] = False

        # Setting all steps in the forecast window to be able to pay attention to steps in the context window
        mask[self.context_window:, :self.context_window] = False

        return mask
    
class PredictionBlock(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.fcn1 = nn.Linear(input_channels, output_channels)
        self.fcn2 = nn.Linear(output_channels, output_channels)

    def forward(self, x):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        x = self.fcn1(x)
        x = torch.relu(x)
        x = self.fcn2(x)
        return x

class Model(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, num_heads, embed_dim, context_window, forecast_window, context_dim):
        super().__init__()
        self.spatio_temporal1 = SpatioTemporalBlock(input_channels, hidden_channels, output_channels, num_heads, embed_dim, context_window, forecast_window, context_dim)
        self.prediction = PredictionBlock(hidden_channels, output_channels)
        #self.num_st_iterations = num_st_iterations

    def forward(self, x, edge_index, temporal_context, weather_context, prediction_mask, masked_only = False):
        """
        Parameters:
            x (Tensor): The network nodes and features with shape [num_nodes, num_timesteps]
            edge_index (Tensor): The relationships between nodes
            prediction_mask (Tensor): shape [num_nodes, num_timesteps], True for masked nodes that need predictions.

        Returns
            Tensor: The output embedding with shape [num_nodes, num_timesteps]
        """
        x = self.spatio_temporal1(x, edge_index, temporal_context, weather_context, prediction_mask, masked_only)

        x = self.prediction(x)
        return x

