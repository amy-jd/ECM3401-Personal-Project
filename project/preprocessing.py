import pandas as pd
import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
import torch
from sklearn.model_selection import train_test_split
from itertools import combinations
import hyperparameters as hp
import random
from itertools import combinations
import xarray as xr


pd.set_option('display.max_rows', 500)

#=================================================================================
# Node feature preprocessing
#=================================================================================

def find_long_nan_sections(df, max_gap):
    # create a mask for all of the rows with missing values
    missing_vals = df.isna()
    
    prev_row_missing_vals = missing_vals.shift()

    # find the rows where the value of a sensor changes from nan > value, or value > nan
    transition_rows = missing_vals != prev_row_missing_vals

    # assign an id number to each block of vals
    block_ids = transition_rows.cumsum()

    # find the length of each gap
    gap_lengths = missing_vals.groupby(block_ids).transform('sum')

    # identify all gaps which are longer than 4 hours
    long_gaps = missing_vals & (gap_lengths > max_gap)

    return long_gaps
    
def find_outlier_values(series):
    rolling_med = series.rolling(window=24, center=True).median()
    diff = (series - rolling_med).abs()
    threshold = 5 * diff.rolling(window=24, center=True).median()
    outlier_mask = diff > threshold
    return outlier_mask

def month_based_train_val_test_split(flowdata_df, train_val_test_ratios):
    # get all the months
    df_month_strata = pd.DataFrame(index=flowdata_df.index)
    # Use the index instead of a 'timestamp' column
    df_month_strata['year_month'] = flowdata_df.index.to_period('M')
    print(f"Rows per month: {df_month_strata['year_month'].value_counts().sort_index()}")

    months = df_month_strata['year_month'].unique()
    random.shuffle(months)

    # assign each month to a set
    num_months = df_month_strata['year_month'].nunique()
    train_ratio, val_ratio, test_ratio = train_val_test_ratios

    num_months_train = int(num_months * train_ratio)
    num_months_val = int(num_months * val_ratio)
    num_months_test = num_months - num_months_train - num_months_val

    train_months = months[:num_months_train]
    val_months = months[num_months_train:num_months_train + num_months_val]
    test_months = months[num_months_train + num_months_val:]

    # split the data into the sets
    train_df = flowdata_df[df_month_strata['year_month'].isin(train_months)]
    val_df = flowdata_df[df_month_strata['year_month'].isin(val_months)]
    test_df = flowdata_df[df_month_strata['year_month'].isin(test_months)]

    return train_df, val_df, test_df


def assign_strata(df):
    """
    Takes a timeseries dataframe, and creates a corresponding series specificying each row's strata.
    """

    strata_dict = {
        'part_of_day': {
            'feature_origin': df.index.hour,
            'bins': [0, 5, 9, 13, 17, 20, 24],  
            'labels': ['night', 'early-morning', 'late-morning', 'afternoon', 'early-evening', 'late-evening']
        },
        'part_of_week': {
            'feature_origin': df.index.dayofweek,
            'bins': [0, 5, 7], 
            'labels': ['weekday', 'weekend']
        },
        'part_of_year': {
            'feature_origin': df.index.isocalendar().week,
            'bins': [0, 13, 31, 39, 50, 53], 
            'labels': ['winter', 'spring', 'summer', 'autumn', 'winter']
        }
    }

    strata_df = pd.DataFrame(index=df.index)

    for strata_name, strata_info in strata_dict.items():
        strata_df[strata_name] = pd.cut(
            strata_info['feature_origin'],
            bins=strata_info['bins'],
            labels=strata_info['labels'],
            right=False,  
            include_lowest=True,
            ordered=False
        )

    strata_df['strata'] = strata_df['part_of_day'].astype(str) + '_' + strata_df['part_of_week'].astype(str) + '_' + strata_df['part_of_year'].astype(str)

    return strata_df


def create_samples(df, df_context, sample_length, set_name, overlap):
    """
    Creates samples of the data with a given length and overlap.

    Note: 
    - Currently all samples will start at the same time of day, as each one is 5 days long
    - This would make the model struggle with differently timed inputs
    - I will need to eventually add random starting points for the samples, but for now I will just create the samples with a fixed starting point to test the model
    """

    samples_df = pd.DataFrame(columns=hp.SENSOR_COLS)
    samples_context_df = pd.DataFrame(columns=df_context.columns, index=samples_df.index)

    # Find all the gaps in the data (where there are missing time steps / it is not continuous)
    gap_mask = df.index.to_series().diff() > pd.Timedelta(minutes=15)

    time_diffs = df.index.to_series().diff()

    # Identify actual gaps (where diff > 15 minutes)
    gap_durations = time_diffs[time_diffs > pd.Timedelta(minutes=15)]
    print(f'\nGap summary for {set_name} set:')
    for idx, gap in gap_durations.items():
        prev_time = df.index[df.index.get_loc(idx) - 1]
        print(f"Gap before {idx}: {gap} (from {prev_time} to {idx})")

    # Split the data into all of the continous segments
    df = df.copy()
    df['segment_id'] = gap_mask.cumsum()
    segments_df = df.groupby('segment_id')

    print(f'Segments in {set_name} set: {len(segments_df)}')

    if overlap:
            step = sample_length // 2
    else:
        step = sample_length

    # Split the data into all of the continuous segments and iterate through each segment
    for _, segment in segments_df:

        print(f'Processing segment {segment["segment_id"].iloc[0]} in {set_name} set, length: {len(segment)}, from {segment.index[0]} to {segment.index[-1]}')

        # Get rid of the segment_id column
        segment = segment.drop(columns='segment_id')

        context_segment = df_context.loc[segment.index]

        num_samples = 0
        i = 0
        while i + sample_length <= len(segment):
            index = len(samples_df)

            sample_row = {}
            context_row = {}
            sample = segment[i:i + sample_length]
            context_sample = context_segment[i:i + sample_length]

            for col in hp.SENSOR_COLS:
                sample_row[col] = sample[col].values

            for context_col in context_segment.columns:
                context_row[context_col] = context_sample[context_col].values

            samples_df.loc[index] = sample_row
            samples_context_df.loc[index] = context_row

            i += step
            num_samples += 1
        
        print(f'Created {num_samples}')
    
    return samples_df, samples_context_df
    

def strat_random_sampling(windows_df, strata_df):
    strata_series = strata_df['strata'].apply(lambda arr: arr[0]) 

    counts = strata_series.value_counts()
    min_count = counts.min()

    sampled_idx = strata_series.groupby(strata_series).sample(n=min_count, random_state=42).index

    windows_df_sampled = windows_df.loc[sampled_idx]
    strata_df_sampled = strata_df.loc[sampled_idx]

    return windows_df_sampled, strata_df_sampled

def generate_masks(df, strata_df, forecast_window=hp.FORECAST_WINDOW):

    rows = []
    strata = []
    input_masks = []
    prediction_masks = []

    sensor_cols = hp.SENSOR_COLS
    num_nodes = len(sensor_cols)
    sample_length = hp.TOTAL_WINDOW

    # Generate all combinations of 1-3 nodes to mask
    node_combos = []
    for num_masked in range(1, 4):
        mask_combinations = combinations(range(num_nodes), num_masked)
        for combo in mask_combinations:
            node_combos.append(combo)

    total_variations = len(node_combos) * len(df)
    print(f"Generating {len(node_combos)} masked variations per sample " f"({len(df)} samples, {total_variations} total)")

    for idx in range(len(df)):
        sample_row = df.iloc[idx]
        strata_row = strata_df.iloc[idx]

        # Go through each combo of node mask
        for combo in node_combos:
            masked_node_set = set(combo)

            input_mask_entry = {}
            prediction_mask_entry = {}
            # Go through each column in the sample
            for col_idx, col in enumerate(sensor_cols):
                input_mask = np.ones(sample_length)
                prediction_mask = np.ones(sample_length)

                # Mask selected nodes (entire time series)
                if col_idx in masked_node_set:
                    input_mask[:] = 0.0
                    prediction_mask[-forecast_window:] = 0.0
                else:
                    input_mask[-forecast_window:] = 0.0

                input_mask_entry[col] = input_mask
                prediction_mask_entry[col] = prediction_mask

            rows.append(sample_row.to_dict())
            strata.append(strata_row.to_dict())
            input_masks.append(input_mask_entry)
            prediction_masks.append(prediction_mask_entry)

    rows_df = pd.DataFrame(rows, columns=sensor_cols).reset_index(drop=True)
    input_masks_df = pd.DataFrame(input_masks, columns=sensor_cols).reset_index(drop=True)
    prediction_masks_df = pd.DataFrame(prediction_masks, columns=sensor_cols).reset_index(drop=True)
    strata_df = pd.DataFrame(strata, columns=strata_df.columns).reset_index(drop=True)
    
    return rows_df, strata_df, input_masks_df, prediction_masks_df


def preprocess_flowdata(path, window_size=hp.TOTAL_WINDOW):

    # Reading in the flow data file
    df_flowdata = pd.read_csv(path, index_col=0)
    df_flowdata.index = pd.to_datetime(df_flowdata.index, format='%d/%m/%Y %H:%M')

    # Removing a sensor with a large number of missing values
    df_flowdata = df_flowdata.drop('1615', axis=1)

    df_flowdata = df_flowdata.rename(columns=hp.SENSOR_DMA_TO_ID)
    df_flowdata = df_flowdata.sort_index(axis=1)

    # Removing rows which have outliers or are part of long sections of missing values
    rows_to_remove = pd.Series(False, index=df_flowdata.index) 
    for col in df_flowdata.columns:
        rows_to_remove |= find_long_nan_sections(df_flowdata[col], hp.MAX_GAP)
        #rows_to_remove |= find_outlier_values(df_flowdata[col])
    df_flowdata = df_flowdata[rows_to_remove == False]

    # Imputing short ranges of missing values
    df_flowdata = df_flowdata.interpolate(method='spline', order = 3)

    # Applying a transformation
    df_flowdata = df_flowdata.apply(np.log1p)

    weather_context_df = load_in_weather_data(hp.WEATHERDATA_PATH, region_idx=16)

    flow_datasets = []
    context_datasets = []
    input_masks = []
    prediction_masks = []

    train_df, val_df, test_df = month_based_train_val_test_split(df_flowdata, hp.TRAIN_VAL_TEST_SPLIT)
    dfs = [train_df, val_df, test_df]
    overlap = [True, True, False]

    set_names = ['train', 'val', 'test']


    for i, df in enumerate(dfs):
        set_name = set_names[i]
        strata_df = assign_strata(df)

        context_df = strata_df.join(weather_context_df, how='left')

        samples_df, samples_context_df = create_samples(df, context_df, window_size, set_name, overlap=overlap[i])
        sampled_df, sampled_context_df = strat_random_sampling(samples_df, samples_context_df)

        expanded_flow_df, expanded_context_df, input_masks_df, prediction_masks_df,  = generate_masks(sampled_df, sampled_context_df)

        flow_datasets.append(expanded_flow_df)
        input_masks.append(input_masks_df)
        prediction_masks.append(prediction_masks_df)
        context_datasets.append(expanded_context_df)

        print(f'strata df length: {len(strata_df)}, context df length: {len(context_df)}')

   
    return flow_datasets, context_datasets, list(zip(input_masks, prediction_masks))
#=================================================================================
# Graph preprocessing
#=================================================================================


def load_graph(file_path):

    df_subsetgraph = pd.read_csv(file_path)
    df_node_items = df_subsetgraph[df_subsetgraph['SensorIndicator'] == 1]

    g = nx.Graph()
    for i, row1 in df_node_items.iterrows():
        for j, row2 in df_node_items.iterrows():
            if row1['SensorDMA'] > row2['SensorDMA']:
                g.add_edge(str(int(row1['SensorDMA'])), str(int(row2['SensorDMA'])), weight=calc_edge_weight('euclidean', row1, row2))
            
    return g


def calc_edge_weight(technique, row1, row2):
    if technique == 'euclidean':
        return euclidean(row1[['XMid', 'YMid']], row2[['XMid', 'YMid']])
    else:
        raise ValueError(f"Unknown technique: {technique}")
    

def assign_node_indices(g):
    node_list = list(g.nodes)
    node_to_index = {}

    for i, node in enumerate(node_list):
        node_to_index[node] = i

    return node_to_index

def get_edge_information(g, node_to_index):
    indexed_edge_start_nodes = []
    indexed_edge_end_nodes = []
    edge_weights = []

    edge_list = g.edges(data=True)

    for start_node, end_node, weight in edge_list:
        indexed_edge_start_nodes.append(hp.SENSOR_DMA_TO_ID[start_node])
        indexed_edge_end_nodes.append(hp.SENSOR_DMA_TO_ID[end_node])
        edge_weights.append(weight['weight'])

    return [indexed_edge_start_nodes, indexed_edge_end_nodes], edge_weights

def preprocess_graph(path):

    # Abstract the flow network dataset, into a graph representing sensors as connected nodes
    g = load_graph(path)

    g.remove_node('1615')

    # Assign each node an index number
    node_to_index = assign_node_indices(g)

    # Flattens the graph into an edge list and list of corresponding edge weights
    edge_index, edge_weight = get_edge_information(g, node_to_index)

    # Converts the lists to tensors
    edge_index_tensor = torch.tensor(edge_index)
    edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.long)

    return edge_index_tensor, edge_weight_tensor


#=================================================================================
# Weather data preprocessing
#=================================================================================

def load_in_weather_data(path, region_idx=16):
    ds = xr.open_dataset(path, engine='netcdf4')

    start_date = '2020-12-31'
    end_date = '2024-09-29'

    ds_subset = ds.sel(region=region_idx, time=slice(start_date, end_date))

    df = ds_subset.to_dataframe().reset_index()


    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.drop(['region', 'geo_region', 'time_bnds', 'bnds'], axis=1)
    df = df.drop_duplicates()

    df = df.resample("15min").ffill()

    cut_off_time = df.index.min() + pd.Timedelta(hours=12)
    df = df[df.index >= cut_off_time]

    return df
