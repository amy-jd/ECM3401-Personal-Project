import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
import torch
from sklearn.model_selection import train_test_split
from eda import plotGraph




pd.set_option('display.max_rows', 500)

INPUT_WINDOW = 4
FORECAST_WINDOW = 8
TOTAL_WINDOW = INPUT_WINDOW + FORECAST_WINDOW
MAX_GAP = 16

#SENSOR_COLS = ['919', '157', '1959', '1016', '1994', '1870']
SENSOR_COLS = [0,1,2,3,4,5]
SENSOR_DMA_TO_ID = {
    '919': 0, 
    '157': 1, 
    '1016': 2, 
    '1870': 3,
    '1959': 4, 
    '1994': 5,
}



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

def assign_strata(df_flowdata):

    strata_dict = {
        'time_of_day': {
            'feature_origin': df_flowdata.index.hour,
            'bins': [0, 6, 12, 14, 18, 22, 24],  
            'labels': ['night', 'morning', 'midday', 'afternoon', 'evening', 'night']
        },
        'part_of_week': {
            'feature_origin': df_flowdata.index.dayofweek,
            'bins': [0, 5, 7], 
            'labels': ['weekday', 'weekend']
        },
        'season': {
            'feature_origin': df_flowdata.index.month,
            'bins': [0, 3, 6, 9, 12, 13], 
            'labels': ['winter', 'spring', 'summer', 'autumn', 'winter']
        }
    }

    for strata_name, strata_info in strata_dict.items():
        df_flowdata[strata_name] = pd.cut(
            strata_info['feature_origin'],
            bins=strata_info['bins'],
            labels=strata_info['labels'],
            right=False,  
            include_lowest=True,
            ordered=False
        )

    df_flowdata['strata'] = df_flowdata['time_of_day'].astype(str) + '_' + df_flowdata['part_of_week'].astype(str) + '_' + df_flowdata['season'].astype(str)

    return df_flowdata


def create_samples(df_flowdata):

    gap_mask = df_flowdata.index.to_series().diff() > pd.Timedelta(minutes=15)
    df_flowdata['segment_id'] = gap_mask.cumsum()

    windows_df = pd.DataFrame(columns=SENSOR_COLS)
    strata_series = pd.Series(dtype='object', name='strata')

    for _, segment in df_flowdata.groupby('segment_id'):
        segment = segment.drop(columns='segment_id')

        sensor_values = segment[SENSOR_COLS].values
        strata_values = segment['strata'].values

        i = 0
        while i + TOTAL_WINDOW <= len(segment):
            row = {
                col: sensor_values[i:i + TOTAL_WINDOW, idx]
                for idx, col in enumerate(SENSOR_COLS)
            }

            index = len(windows_df)

            windows_df.loc[index] = row
            strata_series.loc[index] = strata_values[i]

            i += TOTAL_WINDOW

    return windows_df, strata_series
    
def strat_random_sampling(windows_df, strata_series):
    counts = strata_series.value_counts()
    min_count = counts.min()

    sampled_idx = strata_series.groupby(strata_series).sample(n=min_count, random_state=42).index

    windows_df_sampled = windows_df.loc[sampled_idx]
    strata_series_sampled = strata_series.loc[sampled_idx]

    return windows_df_sampled, strata_series_sampled


def train_val_test_split(windows_df_sampled, strata_series_sampled):

    train_idx, temp_idx = train_test_split(
        windows_df_sampled.index,
        test_size=0.3,
        stratify=strata_series_sampled.loc[windows_df_sampled.index],
        random_state=42
    )

    train_df = windows_df_sampled.loc[train_idx]
    train_strata = strata_series_sampled.loc[train_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=strata_series_sampled.loc[temp_idx],
        random_state=42
    )

    val_df = windows_df_sampled.loc[val_idx]
    val_strata = strata_series_sampled.loc[val_idx]

    test_df = windows_df_sampled.loc[test_idx]
    test_strata = strata_series_sampled.loc[test_idx]

    
    return [train_df, val_df, test_df], [train_strata, val_strata, test_strata]


def preprocess_flowdata(path):

    # Reading in the flow data file
    df_flowdata = pd.read_csv(path, index_col=0)
    df_flowdata.index = pd.to_datetime(df_flowdata.index, format='%d/%m/%Y %H:%M')

    # Removing a sensor with a large number of missing values
    df_flowdata = df_flowdata.drop('1615', axis=1)

    df_flowdata = df_flowdata.rename(columns=SENSOR_DMA_TO_ID)
    df_flowdata = df_flowdata.sort_index(axis=1)

    # Removing rows which have outliers or are part of long sections of missing values
    rows_to_remove = pd.Series(False, index=df_flowdata.index) 
    for col in df_flowdata.columns:
        rows_to_remove |= find_long_nan_sections(df_flowdata[col], MAX_GAP)
        #rows_to_remove |= find_outlier_values(df_flowdata[col])
    df_flowdata = df_flowdata[rows_to_remove == False]

    # Imputing short ranges of missing values
    df_flowdata = df_flowdata.interpolate(method='spline', order = 3)

    # Applying a transformation
    df_flowdata = df_flowdata.apply(np.log1p)

    # Stratified random sampling
    df_flowdata = assign_strata(df_flowdata)
    windows_df, strata_series = create_samples(df_flowdata)
    windows_df_sampled, strata_series_sampled = strat_random_sampling(windows_df, strata_series)

    # Splitting into train val and test sets
    split_df, split_df_strata = train_val_test_split(windows_df_sampled, strata_series_sampled)
   
    return split_df, split_df_strata


#=================================================================================
# Graph preprocessing

"""
        split_df = {
            'train' : [train_df, train_strata],
            'val' : [val_df, val_strata],
            'test' : [test_df, test_strata]
        }
"""
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
        indexed_edge_start_nodes.append(SENSOR_DMA_TO_ID[start_node])
        indexed_edge_end_nodes.append(SENSOR_DMA_TO_ID[end_node])
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




