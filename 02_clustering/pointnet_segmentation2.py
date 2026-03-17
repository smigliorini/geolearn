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
from torch_geometric.nn import MLP, fps, global_max_pool, knn_interpolate
from torch_geometric.nn import PointNetConv
from torchvision import transforms
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import json
import os
from os.path import exists
from torcheval.metrics import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning) 




def radius(
    x: torch.Tensor,
    y: torch.Tensor,
    r: float,
    batch_x: Optional[torch.Tensor] = None,
    batch_y: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    if x.numel() == 0 or y.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y
    x, y = x.contiguous(), y.contiguous()

    if batch_size is None:
        batch_size = 1
        if batch_x is not None:
            assert x.size(0) == batch_x.numel()
            batch_size = int(batch_x.max()) + 1
        if batch_y is not None:
            assert y.size(0) == batch_y.numel()
            batch_size = max(batch_size, int(batch_y.max()) + 1)
    assert batch_size > 0

    ptr_x: Optional[torch.Tensor] = None
    ptr_y: Optional[torch.Tensor] = None

    if batch_size > 1:
        assert batch_x is not None
        assert batch_y is not None
        arange = torch.arange(batch_size + 1, device=x.device)
        ptr_x = torch.bucketize(arange, batch_x)
        ptr_y = torch.bucketize(arange, batch_y)

    return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
                                          max_num_neighbors, num_workers)


    

max_temp = 699
min_temp = -699
max_label = 28

# cmap = plt.get_cmap('viridis')
# colors = [cmap(float(i)/max_label)[:3] for i in range(max_label+1)]
# _colors_tensor = torch.tensor(colors).to(torch.float32)
# colors_tensor = torch.tensor(colors).to(torch.float32)

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
        y = torch.zeros(input_pos.shape[0], max_label+1)
        # actual_y = torch.zeros(input_pos.shape[0], 1)
        for i in range(max_label+1):
            # y[df['LABEL'] == i, :] = _colors_tensor[i, :]
            y[df['LABEL'] == i, i] = 1
            # actual_y[df['LABEL'] == i, :] = i
        data = Data(pos=input_pos.to(torch.float32),
                    x=input_x.to(torch.float32),
                    y=y.to(torch.float32),
                    # actual_y=actual_y.to(torch.float32)
        )
        self.cache[idx] = data
        return data

# def color_to_label(predictions):
#     distances = torch.cdist(predictions, colors_tensor)
#     return torch.argmin(distances, dim=1)

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
        self,
        set_abstraction_ratio_1, set_abstraction_ratio_2,
        set_abstraction_radius_1, set_abstraction_radius_2, max_neighbors, dropout, n_outputs
    ):
        super().__init__()

        # Input channels account for both `pos` and node features.
        self.sa1_module = SetAbstraction(
            set_abstraction_ratio_1,
            set_abstraction_radius_1,
            max_neighbors,
            MLP([2 + 4, 64, 64, 128]) # 4 = (2 dimensions for the points) + (1 dimensions for the features)
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
        self.fp1_module = FeaturePropagation(3, MLP([128 + 4, 128, 128, 128]))
        # self.fp0_module = FeaturePropagation(16, MLP([128, 128, 128, 128]))


        self.mlp = MLP([128, 128, 128, max_label+1], dropout=dropout, norm="batch_norm") #

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


with open('./config_general.json') as f:
    config = json.load(f)


label = 'general' #sys.argv[1]
with open('./config_%s.json' % label) as f:
    config_exp = json.load(f)
device = (torch.device(config['device']))

batch_size = 8

train_loader = DataLoader(SpatialDatasetSeg(years=train_years), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SpatialDatasetSeg(years=val_years), batch_size=batch_size, shuffle=True)

# colors_tensor = colors_tensor.to(device)

model = PointNetSeg(
    config['set_abstraction_ratio_1'],
    config['set_abstraction_ratio_2'],
    config['set_abstraction_radius_1'],
    config['set_abstraction_radius_2'],
    config['max_neighbors'],
    config['dropout'],
    1,
).to(device)

for name, layer in model._modules.items():
    print("#####" + name + "####")
    for name, _layer in layer._modules.items():
        print(name, _layer)

# Define Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=config['learning_rate']
)



def train_step(epoch):
    """Training Step"""
    model.train()
    epoch_loss = 0.0
    metric = MulticlassAccuracy(average='micro', num_classes=max_label+1)
    num_batches = len(train_loader)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{config['epochs']}"
    )
    iterator = iter(train_loader)
    for batch_idx in progress_bar:
        data = next(iterator).to(device)
        # actual_y = data.actual_y
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.cross_entropy(prediction, data.y)
        # prediction = color_to_label(prediction)
        metric.update(torch.argmax(prediction, dim=1).flatten(), torch.argmax(data.y, dim=1).flatten())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_batches
    acc = metric.compute()
    print('train loss: %f, train acc: %s' % ( epoch_loss, str(acc) ))

def val_step(epoch, best_accuracy):
    """Validation Step"""
    model.eval()
    epoch_loss = 0.0
    num_batches = len(val_loader)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Validation Epoch {epoch}/{config['epochs']}"
    )
    actual = None
    predictions = None
    batches = None
    metric1 = MulticlassAccuracy(average='micro', num_classes=max_label+1)
    metric2 = MulticlassPrecision(average='micro', num_classes=max_label+1)
    metric3 = MulticlassRecall(average='micro', num_classes=max_label+1)
    metric4 = MulticlassF1Score(average='micro', num_classes=max_label+1)
    iterator = iter(val_loader)
    for batch_idx in progress_bar:
        data = next(iterator).to(device)
        actual = data.y if actual == None else torch.cat((actual, data.y))
        batches = data.batch + (batch_idx * batch_size) if batches == None else torch.cat((batches, data.batch + (batch_idx * batch_size)))
        with torch.no_grad():
            prediction = model(data)
        loss = F.cross_entropy(prediction, data.y)
        metric1.update(torch.argmax(prediction, dim=1), torch.argmax(data.y, dim=1))
        metric2.update(torch.argmax(prediction, dim=1), torch.argmax(data.y, dim=1))
        metric3.update(torch.argmax(prediction, dim=1), torch.argmax(data.y, dim=1))
        metric4.update(torch.argmax(prediction, dim=1), torch.argmax(data.y, dim=1))
        
        predictions = prediction.cpu() if predictions == None else torch.cat((predictions.cpu(), prediction.cpu()))
        epoch_loss += loss.item()
    m1 = metric1.compute()
    m2 = metric2.compute()
    m3 = metric3.compute()
    m4 = metric4.compute()
    #wmape =   weighted_mean_absolute_percentage_error(predictions.cpu(), actual.cpu())
    epoch_loss = epoch_loss / num_batches
    print('val loss: %f\t accuracy: %f\t precision: %f\t recall: %f\t f1: %f' % (epoch_loss, m1, m2, m3, m4))
    if m4 > best_accuracy:
        cur_path = config["output_folder"] + "/pointnet_segmentation_parametrized_%.6f.csv" % (best_accuracy)
        checkpoint_path = config["output_folder"] + "/pointnet_segmentation_parametrized_%.6f.ckpt" % (best_accuracy)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        # best_wampe = wmape
        best_accuracy = m4
        df = pd.DataFrame()
        df["batches"] = batches.cpu().numpy()
        df["actual"] = torch.argmax(actual, dim=1).flatten().cpu().numpy()
        df["predicted"] = torch.argmax(predictions, dim=1).flatten().cpu().numpy()
        
        # df["partition_label"] = labels
        cur_path = config["output_folder"] + "/pointnet_segmentation_parametrized_%.6f.csv" % (best_accuracy)
        checkpoint_path = config["output_folder"] + "/pointnet_segmentation_parametrized_%.6f.ckpt" % (best_accuracy)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)
    return best_accuracy


        
best_accuracy = 0
best_wmape = 100
best_loss = 1e10
total_time = 0

for epoch in range(1, config['epochs'] + 1):
    start = timer()
    train_step(epoch)
    total_time += timer() - start
    best_accuracy = val_step(epoch, best_accuracy)
