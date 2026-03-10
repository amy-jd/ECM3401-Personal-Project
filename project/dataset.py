import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import hyperparameters as hp

class WaterFlowDataSet(Dataset):

    def __init__(self, df, df_strata, edge_index, edge_weight, forecast_window=0, generate_masks=True, masks=None):
        self.df = df
        self.df_strata = df_strata

        #self.input_masks_df, self.prediction_masks_df = df_masks

        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.forecast_window = forecast_window
        self.generate_masks = generate_masks
        self.node_names = list(df.columns)
        self.N = len(self.node_names)

        if masks is not None:
            self.input_masks, self.prediction_masks = masks
        else:
            self.input_masks = None
            self.prediction_masks = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        x = np.stack(row.values, axis=0)
        x = torch.tensor(x, dtype=torch.float)


        if self.generate_masks:
            # Generating a mask, where future values and masked nodes are set to 0.
            input_mask, prediction_mask = self.generate_mask(x)
        else:
            input_mask_row = self.input_masks.iloc[idx]
            input_mask = torch.tensor(np.stack(input_mask_row.values, axis=0), dtype=torch.float)

            prediction_mask_row = self.prediction_masks.iloc[idx]
            prediction_mask = torch.tensor(np.stack(prediction_mask_row.values, axis=0), dtype=torch.float)


        # Apply the mask to the input data, so that masked positions are set to 0 and unmasked positions retain their original values
        x_masked = x * input_mask
        # Transforming the mask to be a boolean tensor, where masked positions (0) are True, and unmasked positions (1) are False
        boolean_mask = (input_mask == 0) 
        boolean_prediction_mask = (prediction_mask == 0)

        #edge_index_tensor = torch.tensor(self.edge_index)
        #edge_weight_tensor = torch.tensor(self.edge_weight, dtype=torch.long)

        strata_row = self.df_strata.iloc[idx]
        context_tensor = self.generate_time_context_tensor(strata_row)

        data = Data(
            x = x_masked,
            y = x,       
            input_mask = boolean_mask,
            prediction_mask = boolean_prediction_mask,            
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
            Tensor: A mask tensor of the same shape as x, where masked positions are 0 and unmasked positions are 1
        """

        # Create a mask of shape [num_nodes, num_timesteps] initialized all 1s (unmasked)
        mask = torch.ones_like(x)
        prediction_mask = torch.ones_like(x)

        # Select 1-3 nodes, which will be masked
        num_nodes_to_mask = torch.randint(1, 4, (1,)).item()
        node_indices = torch.randperm(self.N)[:num_nodes_to_mask]

        # Mask the nodes, setting all their time steps to 0 (masked)
        mask[node_indices, :] = 0.0  

        # Mask future nodes for forecasting, setting the 'forecast_window' time steps for all nodes to 0 (masked)
        mask[:, -self.forecast_window:] = 0.0

        prediction_mask[node_indices, -self.forecast_window:] = 0.0

        return mask, prediction_mask
    
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
            
    

