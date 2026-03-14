import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import hyperparameters as hp

class WaterFlowDataSet(Dataset):

    def __init__(self, df, context_df, edge_index, edge_weight, forecast_window=0, generate_masks=True, masks=None):
        self.df = df
        self.context_df = context_df

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

        context_row = self.context_df.iloc[idx]

        time_context = context_row[['part_of_day', 'part_of_week', 'part_of_year']]
        time_context_tensor = self.generate_context_tensor(time_context, context_type='strata', data_type=torch.long)

        weather_context = context_row[hp.WEATHER_COLS]
        weather_context_tensor = self.generate_context_tensor(weather_context, context_type='weather', data_type=torch.float)
        weather_context_tensor = weather_context_tensor * input_mask.unsqueeze(-1)


        data = Data(
            x = x_masked, # (num_nodes, num_timesteps) = (6,430)
            y = x,       
            input_mask = boolean_mask,
            prediction_mask = boolean_prediction_mask,            
            edge_index = self.edge_index,
            edge_weight = self.edge_weight,
            temporal_context = time_context_tensor, 
            weather_context = weather_context_tensor,
            x_gnn = x_masked[:, :hp.CONTEXT_WINDOW] ,
            y_gnn = x[:, :hp.CONTEXT_WINDOW] ,
            prediction_mask_gnn = boolean_mask[:, :hp.CONTEXT_WINDOW] 
        )
        return data
    
    
    def generate_mask(self, x):
        """
        Generates a mask for the input data, where certain nodes and future time steps are masked.

        Parameters:
            x (Tensor): The input tensor of shape [num_nodes, num_timesteps]

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
    
    def generate_context_tensor(self, row, context_type, data_type):
        """
        Parameters:
            strata_row (Series): The strata for a given sample, which specifies its time of day, day of week and season

        Returns:
            Tensor: A tensor representing the time context of the sample, with shape [num_strata_types, num_timesteps]
        """

        all_context = []

        # iterate through each column / strata type in the df
        for col_name in row.index:
            context = []
            # iterate through each value in a sample strata
            for val in row[col_name]:
                if context_type == 'strata':
                    # convert from string to index, making it easier for the model to process
                    val_index = hp.STRATA_TO_INDEX[col_name][val]
                    context.append(val_index)
                else:
                    context.append(val)

            all_context.append(context)

        context_array = np.array(all_context).T # Transpose to get shape [num_timesteps, num_strata_types]

        context_tensor = torch.tensor(context_array, dtype=data_type) # Convert to tensor with shape [num_timesteps, num_strata_types]
 
        context_tensor = context_tensor.unsqueeze(0).repeat(self.N, 1, 1) # Reshape to [num_nodes, num_timesteps, num_strata_types] by adding a node dimension and repeating the context for each node

        return context_tensor
    

            
    

