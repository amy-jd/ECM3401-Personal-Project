import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import hyperparameters as hp

class WaterFlowDataSet(Dataset):

    def __init__(self, df, df_strata, edge_index, edge_weight, forecast_window=0):
        self.df = df
        self.df_strata = df_strata
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        #self.T = num_timesteps
        #self.mask_prob_nodes = mask_prob_nodes
        self.forecast_window = forecast_window
        self.node_names = list(df.columns)
        self.N = len(self.node_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        #TODO: is this tensor the right way round or should i .T it
        row = self.df.iloc[idx]
        x = np.stack(row.values, axis=0)
        x = torch.tensor(x, dtype=torch.float)

        mask = self.generate_mask(x)
        x_masked = x * mask

        #edge_index_tensor = torch.tensor(self.edge_index)
        #edge_weight_tensor = torch.tensor(self.edge_weight, dtype=torch.long)

        strata_row = self.df_strata.iloc[idx]
        #context_tensor = self.generate_context_tensor(strata_row)

        data = Data(
            x = x_masked,
            y = x,       
            mask = (mask == 0),            
            edge_index = self.edge_index,
            edge_weight = self.edge_weight,
            context = 1
        )
        return data
    
    def generate_mask(self, x):

        mask = torch.ones_like(x)

        # Randomly mask 1-3 nodes
        num_nodes_to_mask = torch.randint(1, 4, (1,)).item()
        node_indices = torch.randperm(self.N)[:num_nodes_to_mask]
        mask[node_indices, :] = 0.0  

        # Mask future nodes for forecasting
        if self.forecast_window > 0:
            mask[:, -self.forecast_window:] = 0.0

        return mask
    
#def generate_context_tensor(self, strata_row):
    

