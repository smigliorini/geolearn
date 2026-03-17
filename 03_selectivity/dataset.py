from torch_geometric.data import Data, Dataset
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import torch
from sklearn.neighbors import KDTree
import random

def get_histogram(data):
    histogram = np.zeros((4,128,128))
    for (_, row) in data.iterrows():
        i = int(row['i0'])
        j = int(row['i1'])
        histogram[0, i, j] = row['num_features']
        histogram[1, i, j] = row['avg_area']
        histogram[2, i, j] = row['avg_side_length_0']
        histogram[3, i, j] = row['avg_side_length_1']
    return histogram

class SpatialDataset(Dataset):
    def __init__(self, is_train=True, histogram=False):
        super().__init__()
        self.cache = {}
        self.histogram = histogram
        range_queries = pd.read_csv('/rhome/msaee007/PointNet/other_files/sj_histograms/rangeQueries/rq_small.csv')
        range_queries = range_queries[~range_queries['datasetName'].str.contains('gaussian_020')] # becaues this doesn't have histogram available in the source

        labels = pd.read_csv('/rhome/msaee007/PointNet/other_files/sj_histograms/rangeQuerResults/result_small.csv',sep=';')
        labels = labels[~labels['dataset_numQuery_areaInt'].str.contains('gaussian_020')]

        self.range_queries = range_queries.drop(range_queries.index[::5])
        self.labels = labels.drop(range_queries.index[::5])
        self.labels_max = self.labels['cardinality'].max()
        self.labels_min = self.labels['cardinality'].min()
        self.labels_range = self.labels_max - self.labels_min
        if not is_train:
            self.range_queries = range_queries.iloc[::5]
            self.labels = labels.iloc[::5]
        
    def __len__(self):
        return self.len()
    def __getitem__(self, idx):
        return self.get(idx)

    def len(self):
        return len(self.range_queries)
    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        row = self.range_queries.iloc[idx]
        query = row[['minX','minY','maxX','maxY']].values.astype(float)
        #label = (self.labels.iloc[idx]['cardinality'] - self.labels_min) / self.labels_range
        label = self.labels.iloc[idx]['cardinality']
        dataset_name = row['datasetName'][:-2]
        data_rectangles = pd.read_csv('/rhome/msaee007/bigdata/sj_histograms/datasets/small_datasets/%s.csv' % dataset_name, header=None)
        y = torch.tensor([[label/data_rectangles.shape[0]]]).to(torch.float32)
        if not self.histogram:
            x = torch.zeros((data_rectangles.shape[0], 8))
            x[:, :4] = torch.from_numpy(data_rectangles.values)
            x[:, 4:] = torch.from_numpy(query)
            points = torch.stack(((x[:, 0] + x[:, 2]) / 2, (x[:, 1] + x[:, 3]) / 2), axis=-1)
            data = Data(
                pos=points.to(torch.float32),
                x=x.to(torch.float32),
                y=y
            )
            self.cache[idx] = data
            return data
        else: 
            data_summary = pd.read_csv('/rhome/msaee007/bigdata/sj_histograms/histograms/small_datasets/%s_summary.csv' % dataset_name)
            hist = get_histogram(data_summary)
            out = (torch.from_numpy(hist).to(torch.float32),
                   torch.from_numpy(query).to(torch.float32),
                    y)
            self.cache[idx] = out
            return out