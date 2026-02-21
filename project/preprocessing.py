import pandas as pd
import torch
import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
import torch
from sklearn.model_selection import train_test_split
import hyperparameters as hp
import random

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

def assign_strata(df):
    """
    Takes a timeseries dataframe, and creates a corresponding series specificying each row's strata.
    """

    strata_dict = {
        'time_of_day': {
            'feature_origin': df.index.hour,
            'bins': [0, 6, 12, 14, 18, 22, 24],  
            'labels': ['night', 'morning', 'midday', 'afternoon', 'evening', 'night']
        },
        'part_of_week': {
            'feature_origin': df.index.dayofweek,
            'bins': [0, 5, 7], 
            'labels': ['weekday', 'weekend']
        },
        'season': {
            'feature_origin': df.index.month,
            'bins': [0, 3, 6, 9, 12, 13], 
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

    strata_df['strata'] = strata_df['part_of_week'].astype(str) + '_' + strata_df['season'].astype(str)

    return strata_df


def create_samples(df, df_strata, sample_length, set_name, overlap):
    """
    Creates samples of the data with a given length and overlap.

    Note: 
    - Currently all samples will start at the same time of day, as each one is 5 days long
    - This would make the model struggle with differently timed inputs
    - I will need to eventually add random starting points for the samples, but for now I will just create the samples with a fixed starting point to test the model
    """

    samples_df = pd.DataFrame(columns=hp.SENSOR_COLS)
    samples_strata_df = pd.Series(dtype=str, index=samples_df.index)

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

        strata_segment = df_strata.loc[segment.index]

        num_samples = 0
        i = 0
        while i + sample_length <= len(segment):
            index = len(samples_df)

            sample_row = {}
            strata_row = {}
            sample = segment[i:i + sample_length]
            strata_sample = strata_segment[i:i + sample_length]

            for col in hp.SENSOR_COLS:
                sample_row[col] = sample[col].values
            strata_row = strata_sample.values

            samples_df.loc[index] = sample_row
            samples_strata_df.loc[index] = strata_row

            i += step
            num_samples += 1
        
        print(f'Created {num_samples}')
        

    
    return samples_df, samples_strata_df
    
def strat_random_sampling(windows_df, strata_series):
    counts = strata_series.value_counts()
    min_count = counts.min()

    sampled_idx = strata_series.groupby(strata_series).sample(n=min_count, random_state=42).index

    windows_df_sampled = windows_df.loc[sampled_idx]
    strata_series_sampled = strata_series.loc[sampled_idx]

    return windows_df_sampled, strata_series_sampled


def train_val_test_split(windows_df_sampled, strata_series_sampled):

    train_size = hp.TRAIN_VAL_TEST_SPLIT[0]
    temp_size = 1 - train_size
    val_size = hp.TRAIN_VAL_TEST_SPLIT[1] / temp_size
    test_size = hp.TRAIN_VAL_TEST_SPLIT[2] / temp_size

    train_idx, temp_idx = train_test_split(
        windows_df_sampled.index,
        test_size=temp_size,
        stratify=strata_series_sampled.loc[windows_df_sampled.index],
        random_state=42
    )

    train_df = windows_df_sampled.loc[train_idx]
    train_strata = strata_series_sampled.loc[train_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_size,
        stratify=strata_series_sampled.loc[temp_idx],
        random_state=42
    )

    val_df = windows_df_sampled.loc[val_idx]
    val_strata = strata_series_sampled.loc[val_idx]

    test_df = windows_df_sampled.loc[test_idx]
    test_strata = strata_series_sampled.loc[test_idx]

    
    return [train_df, val_df, test_df], [train_strata, val_strata, test_strata]


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

    datasets = []
    strata = []

    train_df, val_df, test_df = month_based_train_val_test_split(df_flowdata, hp.TRAIN_VAL_TEST_SPLIT)
    dfs = [train_df, val_df, test_df]
    overlap = [True, True, False]

    set_names = ['train', 'val', 'test']

    for i, df in enumerate(dfs):
        set_name = set_names[i]
        strata_df = assign_strata(df)
        samples_df, samples_strata_df = create_samples(df, strata_df, window_size, set_name, overlap=overlap[i])

        datasets.append(samples_df)
        strata.append(samples_strata_df)

   
    return datasets, strata


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




