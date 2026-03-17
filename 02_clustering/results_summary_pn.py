import torch
import json
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import random
from glob import glob
from tqdm.auto import tqdm
from timeit import default_timer as timer
import gc
import torch
import torch.nn.functional as F
import sys
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, knn_interpolate, radius
from torch_geometric.nn import PointNetConv
from torchvision import transforms
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import json
import os
from os.path import exists
from torcheval.metrics import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import time

max_temp = 699
min_temp = -699
max_label = 8
max_label2 = 28

val_years   = ['2000', '1824', '1866', '1928', '1900', '1927', '1858', '1974', '1878', '1909', '1977', '1952', '1881', '1842', '1982',
               '1862', '1916', '1953', '1979', '2015', '1919', '1958', '1965', '1948', '1841', '1929', '2002', '1852', '1985', '2018',
               '1960', '1922', '2016', '1914', '1835', '1876', '1999', '1838', '1860', '1971']

class SpatialDatasetSeg(Dataset):
    def __init__(self, input_folder='/rhome/msaee007/bigdata/weather_data/monthly_labeled/', years=None):
        super().__init__()
        files = glob(input_folder + '*.csv')
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
        file = self.files[idx]
        df = df = pd.read_csv(file)
        input_pos = torch.from_numpy(df[['LONGITUDE', 'LATITUDE']].values)
        input_x = torch.from_numpy((df['TEMP'].values - min_temp) / (max_temp-min_temp))
        input_x = input_x.reshape((input_x.shape[0], 1))
        y = torch.zeros(input_pos.shape[0], max_label)
        for i in range(max_label):
            y[df['LABEL'] == i, i] = 1
        data = Data(pos=input_pos.to(torch.float32),
                    x=input_x.to(torch.float32),
                    y=y.to(torch.float32))
        return data


class SpatialDatasetSeg2(Dataset):
    def __init__(self, input_folder='/rhome/msaee007/bigdata/weather_data/labels_parametrized/', years=None):
        super().__init__()
        self.cache = {}
        files = []
        for p in ['0.05_200_50', '0.02_200_5', '0.03_50_5', '0.05_100_50', '0.03_200_20']:
            files += glob(input_folder + p + '/*.csv')
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
        parts = file.rsplit('/', 2)
        params = [float(num) for num in parts[-2].split('_')]
        df = df = pd.read_csv(file)
        input_pos = torch.from_numpy(df[['LONGITUDE', 'LATITUDE']].values)
        temp = torch.from_numpy((df['TEMP'].values - min_temp) / (max_temp-min_temp))
        input_x = torch.zeros((temp.shape[0], 4))
        input_x[:, 0] = temp
        input_x[:, 1] = params[0]
        input_x[:, 2] = params[1]
        input_x[:, 3] = params[2]
        # input_x = input_x.reshape((input_x.shape[0], 1))
        y = torch.zeros(input_pos.shape[0], max_label2+1)
        # actual_y = torch.zeros(input_pos.shape[0], 1)
        for i in range(max_label2+1):
            # y[df['LABEL'] == i, :] = _colors_tensor[i, :]
            y[df['LABEL'] == i, i] = 1
            # actual_y[df['LABEL'] == i, :] = i
        data = Data(pos=input_pos.to(torch.float32),
                    x=input_x.to(torch.float32),
                    y=y.to(torch.float32),
                    # actual_y=actual_y.to(torch.float32)
                    params = torch.tensor(params).repeat(y.shape[0]).reshape(y.shape[0], len(params)).to(torch.float32)
        )
        self.cache[idx] = data
        return data


class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, max_neighbors, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.max_neighbors = max_neighbors
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = list(range(pos.shape[0])) #if self.ratio == 1.0 else fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos, self.r, batch, batch[idx],
                        max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class GlobalSetAbstraction(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), pos.shape[1]))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class FeaturePropagation(torch.nn.Module):
    # https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip



class PointNetSeg(torch.nn.Module):
    def __init__(
        self, input_features,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, max_neighbors, dropout, n_outputs
    ):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            max_neighbors,
            MLP([2 + input_features, 64, 64, 128]) # 4 = (2 dimensions for the points) + (1 dimensions for the features)
        )
        self.sa2_module = SetAbstraction(
            set_abstraction_ratio_2,
            set_abstraction_radius_2,
            max_neighbors,
            MLP([128 + 2, 128, 128, 256]) # 128 dimensions for the features + 2 dimensions for the points
        )
        self.sa3_module = GlobalSetAbstraction(MLP([256 + 2, 256, 512, 1024])) # 256 dimensions for the features + 2 dimensions for the points
        self.fp3_module = FeaturePropagation(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FeaturePropagation(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FeaturePropagation(3, MLP([128 + input_features, 128, 128, 128]))
        # self.fp0_module = FeaturePropagation(16, MLP([128, 128, 128, 128]))
        self.mlp = MLP([128, 128, 128, n_outputs], dropout=dropout, norm="batch_norm") #
    def forward(self, input_data):
        # print(input_data.x.shape, input_data.pos.shape, input_data.y.shape)
        sa0_out = (input_data.x, input_data.pos, input_data.batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        # x, _, _ = self.fp0_module(*fp1_out, *pred_data)
        return F.log_softmax(self.mlp(x), dim=1)



# load models

pointnet_not_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/dbscan_global_0.889036.ckpt'
pointnet_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/pointnet_segmentation_parametrized_0.557927.ckpt'
unet_not_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/unet_global_0.856794.ckpt'
unet_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/unet_global_0.580316.ckpt'

results_folder = '/rhome/msaee007/PointNet/02_clustering/clustering_results'

with open('./config_general.json') as f:
    config = json.load(f)


label = 'general' #sys.argv[1]
with open('./config_%s.json' % label) as f:
    config_exp = json.load(f)

device = (torch.device(config['device']))

batch_size = 8
val_loader = DataLoader(SpatialDatasetSeg(years=val_years), batch_size=batch_size, shuffle=False)
iter1 = iter(val_loader)
val_loader2 = DataLoader(SpatialDatasetSeg2(years=val_years), batch_size=batch_size, shuffle=False)
iter2 = iter(val_loader2)


pointnet_not_param_model = PointNetSeg(
    1,
    config['set_abstraction_ratio_1'],
    config['set_abstraction_ratio_2'],
    config['set_abstraction_radius_1'],
    config['set_abstraction_radius_2'],
    config['max_neighbors'],
    config['dropout'],
    8,
).to(device)
pointnet_not_param_model.load_state_dict(torch.load(pointnet_not_param_path, weights_only=True))


model = pointnet_not_param_model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model trainable parameters: ', pytorch_total_params)

model.eval()


pointnet_param_model = PointNetSeg(
    4,
    config['set_abstraction_ratio_1'],
    config['set_abstraction_ratio_2'],
    config['set_abstraction_radius_1'],
    config['set_abstraction_radius_2'],
    config['max_neighbors'],
    config['dropout'],
    29,
).to(device)
pointnet_param_model.load_state_dict(torch.load(pointnet_param_path, weights_only=True))
model2 = pointnet_param_model.to(device)

pytorch_total_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
print('model2 trainable parameters: ', pytorch_total_params)

model2.eval()

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
    data = next(iter1).to(device)
    actual = data.y if actual == None else torch.cat((actual, data.y))
    batches = data.batch + (batch_idx * batch_size) if batches == None else torch.cat((batches, data.batch + (batch_idx * batch_size)))
    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        prediction = model(data)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
    predictions = prediction.cpu() if predictions == None else torch.cat((predictions.cpu(), prediction.cpu()))
print('Clustering not parameterized pointnet time: ', total_time) # milli seconds

df = pd.DataFrame()
df["batches"] = batches.cpu().numpy()
df["actual"] = torch.argmax(actual, dim=1).cpu().numpy()
df["predicted"] = torch.argmax(predictions, dim=1).cpu().numpy()

df.to_csv(results_folder + '/pn_not_param.csv', index=False)


num_batches = len(val_loader2)
progress_bar = tqdm(range(num_batches))
actual = None
predictions = None
batches = None
params = None
ids = None # get ids
total_time = 0
for batch_idx in progress_bar:
    data = next(iter2).to(device)
    actual = data.y if actual == None else torch.cat((actual, data.y))
    params = data.params if params == None else torch.cat((params, data.params))
    batches = data.batch + (batch_idx * batch_size) if batches == None else torch.cat((batches, data.batch + (batch_idx * batch_size)))
    with torch.no_grad():
        torch.cuda.synchronize()
        start_event.record()
        prediction = model2(data)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
        
    predictions = prediction.cpu() if predictions == None else torch.cat((predictions.cpu(), prediction.cpu()))
print('Clustering parameterized pointnet time: ', total_time)

df = pd.DataFrame()
df["batches"] = batches.cpu().numpy()
params = params.cpu().numpy()
df["params_0"] = params[:, 0]
df["params_1"] = params[:, 1]
df["params_2"] = params[:, 2]
df["actual"] = torch.argmax(actual, dim=1).cpu().numpy()
df["predicted"] = torch.argmax(predictions, dim=1).cpu().numpy()

df.to_csv(results_folder + '/pn_param.csv', index=False)
