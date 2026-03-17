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
from collections import defaultdict
from collections import Counter

with open('./config_general.json') as f:
    config = json.load(f)

device = (torch.device(config['device']))

batch_size = 8



    

max_temp = 699
min_temp = -699
max_label = 8
cmap = plt.get_cmap('tab10')
label_colors = np.array([cmap(i)[:3] for i in range(max_label)])
colors_tensor = torch.from_numpy(label_colors).to(torch.float32).to(device)
histogram_size = 64
x_edges = np.linspace(0, 1, histogram_size + 1)
y_edges = np.linspace(0, 1, histogram_size + 1)


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
        # output_histogram_count = np.zeros((histogram_size, histogram_size))

        x_bin_indices = np.digitize(points[:, 0], x_edges) - 1
        y_bin_indices = np.digitize(points[:, 1], y_edges) - 1
        output_histogram_labels = defaultdict(list)
        for i in range(points.shape[0]):
            x_bin = x_bin_indices[i]
            y_bin = y_bin_indices[i]
            if 0 <= x_bin < histogram_size and 0 <= y_bin < histogram_size:
                input_histogram[0, x_bin, y_bin] = min(x[i, 0], input_histogram[0, x_bin, y_bin])
                input_histogram[1, x_bin, y_bin] = max(x[i, 0], input_histogram[1, x_bin, y_bin])
                input_histogram[2, x_bin, y_bin] += 1
                output_histogram_labels[(x_bin,y_bin)].append(labels[i])
        input_histogram[2, :, :] = input_histogram[2, :, :] / x.shape[0]
        for (x_bin, y_bin) in output_histogram_labels:
            most_common = Counter(output_histogram_labels[(x_bin,y_bin)]).most_common(1)[0][0]
            output_histogram[0, x_bin, y_bin] += label_colors[most_common, 0]
            output_histogram[1, x_bin, y_bin] += label_colors[most_common, 1]
            output_histogram[2, x_bin, y_bin] += label_colors[most_common, 2]
        # output_histogram[0] = np.divide(output_histogram[0], output_histogram_count, where=output_histogram_count!=0)
        # output_histogram[1] = np.divide(output_histogram[1], output_histogram_count, where=output_histogram_count!=0)
        # output_histogram[2] = np.divide(output_histogram[2], output_histogram_count, where=output_histogram_count!=0)
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


train_loader = DataLoader(SpatialDatasetSeg(years=train_years), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(SpatialDatasetSeg(years=val_years), batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

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

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters: ', pytorch_total_params)


# Define Optimizer
optimizer = torch.optim.Adam(
    model.parameters(), lr=config['learning_rate']
)



def train_step(epoch):
    """Training Step"""
    model.train()
    epoch_loss = 0.0
    metric = MulticlassAccuracy(average='micro', num_classes=max_label)
    num_batches = len(train_loader)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{config['epochs']}"
    )
    iterator = iter(train_loader)
    for batch_idx in progress_bar:
        input_histogram, output_histogram, points, actual_labels, batch = next(iterator)
        input_histogram, output_histogram, actual_labels  = input_histogram.to(device), output_histogram.to(device), actual_labels.to(device)
        optimizer.zero_grad()
        prediction = model(input_histogram)
        loss = F.mse_loss(prediction, output_histogram)
        predicted_labels = label_points_from_histogram(batch, points, prediction).to(device)
        predicted_labels, actual_labels = predicted_labels.flatten(), actual_labels.flatten()
        metric.update(predicted_labels, actual_labels)
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
    metric1 = MulticlassAccuracy(average='micro', num_classes=max_label)
    metric2 = MulticlassPrecision(average='micro', num_classes=max_label)
    metric3 = MulticlassRecall(average='micro', num_classes=max_label)
    metric4 = MulticlassF1Score(average='micro', num_classes=max_label)
    iterator = iter(val_loader)
    for batch_idx in progress_bar:
        input_histogram, output_histogram, points, actual_labels, batch = next(iterator)
        input_histogram, output_histogram, actual_labels = input_histogram.to(device), output_histogram.to(device), actual_labels.to(device)
        actual = actual_labels if actual == None else torch.cat((actual, actual_labels))
        batches = batch if batches == None else torch.cat((batches, batch))
        with torch.no_grad():
            prediction = model(input_histogram)
        loss = F.mse_loss(prediction, output_histogram)
        predicted_labels = label_points_from_histogram(batch, points, prediction).to(device)
        predicted_labels, actual_labels = predicted_labels.flatten(), actual_labels.flatten()
        metric1.update(predicted_labels, actual_labels)
        metric2.update(predicted_labels, actual_labels)
        metric3.update(predicted_labels, actual_labels)
        metric4.update(predicted_labels, actual_labels)
        predictions = predicted_labels.cpu() if predictions == None else torch.cat((predictions.cpu(), predicted_labels.cpu()))
        epoch_loss += loss.item()
    m1 = metric1.compute()
    m2 = metric2.compute()
    m3 = metric3.compute()
    m4 = metric4.compute()
    #wmape =   weighted_mean_absolute_percentage_error(predictions.cpu(), actual.cpu())
    epoch_loss = epoch_loss / num_batches
    print('val loss: %f\t accuracy: %f\t precision: %f\t recall: %f\t f1: %f' % (epoch_loss, m1, m2, m3, m4))
    if m4 > best_accuracy:
        cur_path = config["output_folder"] + "/unet_global_%.6f.csv" % (best_accuracy)
        checkpoint_path = config["output_folder"] + "/unet_global_%.6f.ckpt" % (best_accuracy)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        # best_wampe = wmape
        best_accuracy = m4
        df = pd.DataFrame()
        df["batches"] = batches.numpy().flatten()
        df["actual"] = actual.cpu().numpy().flatten()
        df["predicted"] = predictions.cpu().numpy().flatten()
        # df["partition_label"] = labels
        cur_path = config["output_folder"] + "/unet_global_%.6f.csv" % (best_accuracy)
        checkpoint_path = config["output_folder"] + "/unet_global_%.6f.ckpt" % (best_accuracy)
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
