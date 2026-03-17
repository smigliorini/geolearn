import torch
import json
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
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
from torchvision import transforms
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import json
import os
from os.path import exists
from torcheval.metrics import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
from torch_geometric.data import Data, Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import time

with open('./config_general.json') as f:
    config = json.load(f)

device = (torch.device(config['device']))

max_temp = 699
min_temp = -699
max_label = 8
max_label2 = 28

cmap = plt.get_cmap('tab10')
label_colors = np.array([cmap(i)[:3] for i in range(max_label)])
colors_tensor = torch.from_numpy(label_colors).to(torch.float32).to(device)
histogram_size = 64
x_edges = np.linspace(0, 1, histogram_size + 1)
y_edges = np.linspace(0, 1, histogram_size + 1)

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
        self.cache = {}
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
        points = torch.from_numpy(df[['LONGITUDE', 'LATITUDE']].values)
        x = torch.from_numpy((df['TEMP'].values - min_temp) / (max_temp-min_temp))
        x = x.reshape((x.shape[0], 1))
        # y = torch.zeros(points.shape[0], max_label)
        # for i in range(max_label):
        #     y[df['LABEL'] == i, i] = 1
        labels = df['LABEL'].values
        # input histogram
        input_histogram = np.zeros((3, histogram_size, histogram_size))
        output_histogram = np.zeros((3, histogram_size, histogram_size))
        output_histogram_count = np.zeros((histogram_size, histogram_size))

        x_bin_indices = np.digitize(points[:, 0], x_edges) - 1
        y_bin_indices = np.digitize(points[:, 1], y_edges) - 1
        for i in range(points.shape[0]):
            x_bin = x_bin_indices[i]
            y_bin = y_bin_indices[i]
            if 0 <= x_bin < histogram_size and 0 <= y_bin < histogram_size:
                input_histogram[0, x_bin, y_bin] = min(x[i, 0], input_histogram[0, x_bin, y_bin])
                input_histogram[1, x_bin, y_bin] = max(x[i, 0], input_histogram[1, x_bin, y_bin])
                input_histogram[2, x_bin, y_bin] += 1
                output_histogram[0, x_bin, y_bin] += label_colors[labels[i], 0]
                output_histogram[1, x_bin, y_bin] += label_colors[labels[i], 1]
                output_histogram[2, x_bin, y_bin] += label_colors[labels[i], 2]
                output_histogram_count[x_bin, y_bin] += 1
        input_histogram[2, :, :] = input_histogram[2, :, :] / x.shape[0]
        output_histogram[0] = np.divide(output_histogram[0], output_histogram_count, where=output_histogram_count!=0)
        output_histogram[1] = np.divide(output_histogram[1], output_histogram_count, where=output_histogram_count!=0)
        output_histogram[2] = np.divide(output_histogram[2], output_histogram_count, where=output_histogram_count!=0)
        batch = np.zeros((points.shape[0], 1))
        batch[:, 0] = idx
        data = (torch.from_numpy(input_histogram).to(torch.float32),
                torch.from_numpy(output_histogram).to(torch.float32),
                points.to(torch.float32),
                torch.from_numpy(labels).to(torch.int32),
                torch.from_numpy(batch).to(torch.int32))
        self.cache[idx] = data
        return data

def label_points_from_histogram(batch, points, histograms):
    labels = torch.zeros((points.shape[0], 1))
    x_bin_indices = np.digitize(points[:, 0], x_edges) - 1
    y_bin_indices = np.digitize(points[:, 1], y_edges) - 1
    cur_batch = -1
    cur_hist = None
    hist_i = -1
    for i in range(points.shape[0]):
        new_batch = batch[i].item()
        if new_batch != cur_batch:
            hist_i += 1
            pixels = histograms[hist_i].permute((1,2,0)).reshape((histogram_size*histogram_size, 3))
            distances = torch.cdist(pixels, colors_tensor)
            closest_color_indices = torch.argmin(distances, dim=1)
            cur_hist = closest_color_indices.reshape((histogram_size,histogram_size))
        cur_batch = new_batch
        x_bin = x_bin_indices[i]
        y_bin = y_bin_indices[i]
        if 0 <= x_bin < histogram_size and 0 <= y_bin < histogram_size:
            labels[i] = cur_hist[x_bin, y_bin]
    return labels


def custom_collate_fn(batch):
    # Split batch items into first `D` items and the rest
    input_histograms = default_collate([item[0] for item in batch])
    output_histograms = default_collate([item[1] for item in batch])
    points = torch.cat([item[2] for item in batch], dim=0)
    actual_label = torch.cat([item[3] for item in batch], dim=0)
    idx_list = torch.cat([item[4] for item in batch], dim=0)
    return input_histograms, output_histograms, points, actual_label, idx_list



# load models

pointnet_not_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/dbscan_global_0.889036.ckpt'
pointnet_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/pointnet_segmentation_parametrized_0.557927.ckpt'
unet_not_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/unet_global_0.856794.ckpt'
unet_param_path = '/rhome/msaee007/bigdata/pointnet_data/outputs/unet_global_0.580316.ckpt'

results_folder = '/rhome/msaee007/PointNet/02_clustering/clustering_results'

batch_size = 32
val_loader = DataLoader(SpatialDatasetSeg(years=val_years), batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
iter1 = iter(val_loader)

import torch
import torch.nn as nn


class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=0):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, dropout=0):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Dropout(dropout),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x



class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=3):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0], 0)
        self.Conv2 = conv_block(filters[0], filters[1], 0.1)
        self.Conv3 = conv_block(filters[1], filters[2], 0.1)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1], 0.1)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0], 0.1)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        d3 = self.Up3(e3)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        return out

# Initialize the U-Net model
model = UNet().to(device)
model.load_state_dict(torch.load(unet_not_param_path, weights_only=True))
model = model.to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model trainable parameters: ', pytorch_total_params)

model.eval()

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
    input_histogram, output_histogram, points, actual_labels, batch = next(iter1)
    actual = actual_labels if actual == None else torch.cat((actual, actual_labels))
    batches = batch if batches == None else torch.cat((batches, batch))
    with torch.no_grad():
        input_histogram = input_histogram.to(device)
        torch.cuda.synchronize()
        start_event.record()
        prediction = model(input_histogram)
        end_event.record()
        torch.cuda.synchronize()
        total_time += start_event.elapsed_time(end_event)
    predicted_labels = label_points_from_histogram(batch, points, prediction).to(device)
    predicted_labels, actual_labels = predicted_labels.flatten(), actual_labels.flatten()
    predictions = predicted_labels.cpu() if predictions == None else torch.cat((predictions.cpu(), predicted_labels.cpu()))
print('Clustering not parameterized unet time: ', total_time)

df = pd.DataFrame()
df["batches"] = batches.cpu().numpy().flatten()
df["actual"] = actual.cpu().numpy().flatten()
df["predicted"] = predictions.cpu().numpy().flatten()

df.to_csv(results_folder + '/unet_not_param.csv', index=False)