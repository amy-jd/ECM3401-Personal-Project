import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial.distance import euclidean
import torch
from sklearn.model_selection import train_test_split


def plotGraph(df):
    df_sample = df.sample(n=96).sort_index()

    df_sample.plot(
        figsize=(40, 6),
        sharex=True
    )
    plt.show()