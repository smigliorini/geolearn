import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import exists
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import pandas as pd
from GeometryEncoder import GeometryEncoder
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, Batch
from torcheval.metrics import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
from glob import glob
import random
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree

output_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/main_exp_outputs'
device = torch.device('cuda')
max_epochs = 150
batch_size=32


label = 'poly2vec_cluster_'


max_temp = 699
min_temp = -699
max_label = 8

train_years = ['1899', '1931', '1925', '1972', '1889', '1830', '1897', '1845', '1864', '1882', '1827', '1917', '1954', '1896', '1905',
               '2006', '1937', '1898', '1904', '2017', '1869', '1915', '1901', '1836', '1968', '1942', '2011', '1828', '1853', '1926',
               '1847', '1990', '1872', '1902', '1829', '2001', '1865', '1935', '1833', '1849', '2022', '1913', '1918', '1888', '1975',
               '1850', '1826', '1984', '1837', '1957', '1921', '1855', '1857', '1911', '1934', '1945', '1831', '1874', '1950', '1839',
               '1973', '1959', '1940', '1988', '1930', '1856', '1923', '1996', '1750', '2020', '1870', '1994', '1983', '1863', '1851',
               '2004', '1991', '1966', '1859', '1879', '1938', '1933', '1932', '1955', '1875', '1894', '1843', '1924', '2003', '2007',
               '1976', '2023', '1993', '1989', '1943', '1906', '1992', '1964', '1910', '1884', '1868', '1970', '1941', '1834', '1986',
               '1939', '1846', '1946', '2013', '1844', '1951', '1912', '1987', '1963', '1893', '1944', '1969', '1891', '2012', '1861',
               '2021', '2024', '1956', '1997', '1873', '1903', '1978', '1980', '1995', '1998', '1936', '1877', '1871', '1947', '1962',
               '1920', '2009', '2010', '1967', '1895', '2014', '1854', '1840', '1832', '1981', '2008', '2019', '2005', '1848', '1825',
               '1949', '1880', '1907', '1908', '1883', '1867', '1961']

val_years   = ['2000', '1824', '1866', '1928', '1900', '1927', '1858', '1974', '1878', '1909', '1977', '1952', '1881', '1842', '1982',
               '1862', '1916', '1953', '1979', '2015', '1919', '1958', '1965', '1948', '1841', '1929', '2002', '1852', '1985', '2018',
               '1960', '1922', '2016', '1914', '1835', '1876', '1999', '1838', '1860', '1971']

print('Train years: %d \t Val years: %d' % (len(train_years), len(val_years)))

class SpatialDatasetSeg(Dataset):
    def __init__(self, input_folder='/rhome/msaee007/bigdata/pointnet_data/weather_data/monthly_labeled/', years=None):
        super().__init__()
        files = glob(input_folder + '*.csv')
        self.cache = {}
        if years != None:
            _files = []
            for y in years:
                for f in files:
                    if y in f:
                        _files.append(f)
            files = _files
        self.files = files
    def __len__(self):
        return self.len()
    def __getitem__(self, idx):
        return self.get(idx)
    def len(self):
        return len(self.files)
    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        file = self.files[idx]
        df = df = pd.read_csv(file)
        all_input_pos = torch.from_numpy(df[['LONGITUDE', 'LATITUDE']].values)
        input_x = torch.from_numpy((df['TEMP'].values - min_temp) / (max_temp-min_temp))
        input_x = input_x.reshape((input_x.shape[0], 1))
        all_y = torch.zeros(all_input_pos.shape[0], max_label)
        indices = sorted(random.sample(range(all_y.shape[0]), 1024))
        input_pos = all_input_pos[indices]
        input_x = input_x[indices]
        for i in range(max_label):
            all_y[df['LABEL'] == i, i] = 1
        y = all_y[indices]
        tree = cKDTree(input_pos)
        _, nearest_indices = tree.query(all_input_pos, k=1)
        self.cache[idx] = input_pos.float(), input_x.float(), y.float(), all_y, nearest_indices
        return self.cache[idx]

class SimpleTransformerModel(nn.Module):
    def __init__(self, geometry_model, input_dim=36, output_dim=9, max_seq_len=1024, nhead=4, num_layers=4, dim_feedforward=128):
        super(SimpleTransformerModel, self).__init__()
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        self.geometry_encoder = geometry_model
        # # Learnable positional encoding
        # self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, input_dim))
        # nn.init.trunc_normal_(self.positional_encoding, std=0.02)  # Optional init

        # Transformer encoder with 4 layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # # Output head
        self.output_head = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_feedforward, output_dim)
        )

    def forward(self, geometries, lengths, dataset_type, features):
        x = torch.zeros((geometries.shape[0], 1024, 36)).to(device)
        for i in range(geometries.shape[0]):
            x[i, :, :32] = self.geometry_encoder(geometries[i], lengths, dataset_type)
            x[i, :, 32:32+features.shape[-1]] = features[i, :, :]
        # x = self.geometry_encoder(geometries, lengths, dataset_type)
        # x = torch.cat((x,features), axis=1)
        x = self.transformer_encoder(x)
        out = self.output_head(x)
        return out



def custom_collate_fn(batch):
    # batch is a list of tuples: [(tensor1, tensor2, string), ...]
    # points, features, y, all_y, nearest_indices
    tensor1_batch = torch.stack([item[0] for item in batch])
    tensor2_batch = torch.stack([item[1] for item in batch])
    tensor3_batch = torch.stack([item[2] for item in batch])
    tensor4_batch = []
    batch_ids = []
    i = 0
    for item in batch:
        tensor4_batch += item[3]
        batch_ids += [i]*len(item[3])
        i+=1
    tensor4_batch = np.array(tensor4_batch)
    batch_ids = np.array(batch_ids)
    tensor5_batch = [item[4] for item in batch]
    
    return tensor1_batch, tensor2_batch, tensor3_batch, tensor4_batch, tensor5_batch, batch_ids

if __name__ == "__main__":
    model_param_path = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/main_exp_outputs/poly2vec_cluster__0.916178.ckpt'
    results_folder = '/rhome/msaee007/PointNet/02_clustering/clustering_results'
    # train_loader = DataLoader(SpatialDatasetSeg(years=train_years), batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(SpatialDatasetSeg(years=val_years), batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate_fn)

    geometry_model = GeometryEncoder(device).to(device)
    model = SimpleTransformerModel(geometry_model, input_dim=36, output_dim=max_label, max_seq_len=1024, nhead=4, num_layers=4, dim_feedforward=128)
    model.load_state_dict(torch.load(model_param_path, weights_only=True))
    model = model.to(device)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('model trainable parameters: ', pytorch_total_params)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )
    iter1 = iter(val_loader)
    num_batches = len(val_loader)
    progress_bar = tqdm(range(num_batches))
    actual = None
    predictions = None
    batches = None
    ids = None # get ids
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time = 0
    for batch_idx in progress_bar:
        points, features, y, all_y, nearest_indices, batch_ids = next(iter1)
        print()
        actual = all_y if not isinstance(actual, np.ndarray) else np.concatenate((actual, all_y))
        batches = batch_ids + (batch_idx * batch_size) if not isinstance(batches, np.ndarray) else np.concatenate((batches, batch_ids + (batch_idx * batch_size)))
        with torch.no_grad():
            torch.cuda.synchronize()
            start_event.record()
            prediction = model(points.to(device), torch.tensor([]).to(device), "points", features.to(device))
            end_event.record()
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)
        prediction = torch.argmax(prediction, dim=-1).cpu().numpy()
        all_pred = []
        for i in range(len(nearest_indices)):
            all_pred.append(prediction[i][nearest_indices[i]])
        all_pred = np.concatenate(all_pred)
        predictions = all_pred if not isinstance(predictions, np.ndarray) else np.concatenate((predictions, all_pred))
    print('Clustering not parameterized pointnet time: ', total_time) # milli seconds

    df = pd.DataFrame()
    df["batches"] = batches
    df["actual"] = actual.argmax(axis=1)
    df["predicted"] = predictions

    df.to_csv(results_folder + '/poly2vec_not_param.csv', index=False)

