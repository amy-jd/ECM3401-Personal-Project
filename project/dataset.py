import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import hyperparameters as hp

class WaterFlowDataSet(Dataset):

    def __init__(self, df, df_strata, df_masks, edge_index, edge_weight):
        self.df = df
        self.df_strata = df_strata

        self.masks_df, self.node_masks_df, self.forecast_masks_df, self.prediction_masks_df = df_masks

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.node_names = list(df.columns)
        self.N = len(self.node_names)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        x = np.stack(row.values, axis=0)
        x = torch.tensor(x, dtype=torch.float)

        mask, node_mask, forecast_mask, prediction_mask = self.generate_mask_tensors(idx)
        x_masked = x * mask

        strata_row = self.df_strata.iloc[idx]
        context_tensor = self.generate_time_context_tensor(strata_row)

        data = Data(
            x = x_masked,
            y = x,       
            mask = (mask == 0),            
            node_mask = (node_mask == 0),
            forecast_mask = (forecast_mask == 0),
            prediction_mask = (prediction_mask == 0),
            edge_index = self.edge_index,
            edge_weight = self.edge_weight,
            context = context_tensor
        )
        return data
    
    
    def generate_mask_tensors(self, idx):
        """
        Parameters:
            idx (int): The index of the sample to generate masks for

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Tensors representing:
                - mask: combined input mask (node + forecast) used to zero out input values
                - node_mask: which nodes are fully masked
                - forecast_mask: which timesteps are in the forecast window
                - prediction_mask: which positions the model should predict (masked nodes only)
        """
        mask_row = self.masks_df.iloc[idx]
        mask = torch.tensor(np.stack(mask_row.values, axis=0), dtype=torch.float)

        node_mask_row = self.node_masks_df.iloc[idx]
        node_mask = torch.tensor(np.stack(node_mask_row.values, axis=0), dtype=torch.float)

        forecast_mask_row = self.forecast_masks_df.iloc[idx]
        forecast_mask = torch.tensor(np.stack(forecast_mask_row.values, axis=0), dtype=torch.float)

        prediction_mask_row = self.prediction_masks_df.iloc[idx]
        prediction_mask = torch.tensor(np.stack(prediction_mask_row.values, axis=0), dtype=torch.float)

        return mask, node_mask, forecast_mask, prediction_mask
    
    def generate_time_context_tensor(self, strata_row):
        """
        Parameters:
            strata_row (Series): The strata for a given sample, which specifies its time of day, day of week and season

        Returns:
            Tensor: A tensor representing the time context of the sample, which can be used as additional input to the model
        """

        all_context = []

        for strata in ['part_of_day', 'part_of_week', 'part_of_year']:
            context = []
            for val in strata_row[strata]:
                val_index = hp.STRATA_TO_INDEX[strata][val]
                context.append(val_index)
            all_context.append(context)

        context_tensor = torch.tensor(all_context, dtype=torch.long)

        return context_tensor
            
    

