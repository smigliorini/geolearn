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
import geopandas as gpd
import torch
import torch.nn.functional as F
import sys
import json
import os
from os.path import exists
from torcheval.metrics import MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
from torch_geometric.data import Data, Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import to_pil_image
with open('./config_general.json') as f:
    config = json.load(f)

device = (torch.device(config['device']))


    

histogram_size = 64
x_edges = np.linspace(0, 1, histogram_size + 1)
y_edges = np.linspace(0, 1, histogram_size + 1)



max_nodes = 100000


class SpatialDatasetSeg(Dataset):
    def __init__(self, labels_df):
        super().__init__()
        self.cache = {}
        # self.files = files
        # labels_df = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/summary.csv')
        self.labels_df = labels_df #labels_df[labels_df['pois'] > 0].reset_index()
    def __len__(self):
        return self.len()
    def __getitem__(self, idx):
        return self.get(idx)
    def len(self):
        return self.labels_df.shape[0]
    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        row = self.labels_df.iloc[idx]
        place = row['place']
        augment = row['augment']
        lng_max = max(row['east'], row['west']) #max(road_data[:, 4].max(), poi_data[:, 0].max())
        lng_min = min(row['east'], row['west'])#min(road_data[:, 4].min(), poi_data[:, 0].min())
        lng_diff = lng_max-lng_min
        lng_center = (lng_max+lng_min)/2.0
        lat_max = max(row['south'], row['north'])#max(road_data[:, 5].max(), poi_data[:, 1].max())
        lat_min = min(row['south'], row['north'])#min(road_data[:, 5].min(), poi_data[:, 1].min())
        lat_diff = lat_max-lat_min
        lat_center = (lat_max+lat_min)/2.0
        base_path = '/rhome/msaee007/bigdata/pointnet_data/walkability_data/%s/%d_%d_' % (place, int(row['i']), int(row['j']))
        rotation_angle = random.randrange(360)
        pois = pd.read_csv(base_path + 'poi.csv')
        poi_pos = torch.zeros((pois.shape[0], 2))
        poi_xs = []
        poi_ys = []
        for i in range(pois.shape[0]):
            p = pois.geometry[i]
            if augment:
                poi_xs.append(float(p[p.find('(')+1:p.rfind(' ')]))
                poi_ys.append(float(p[p.rfind(' ')+1:p.rfind(')')]))
            else:
                poi_pos[i][0] = (float(p[p.find('(')+1:p.rfind(' ')])-lng_min)/lng_diff
                poi_pos[i][1] = (float(p[p.rfind(' ')+1:p.rfind(')')])-lat_min)/lat_diff
        if augment:
            poi_rotated = gpd.points_from_xy(poi_xs, poi_ys).rotate(rotation_angle, origin=(lng_center,lat_center))
            poi_pos[:, 0] = (torch.tensor([p.x for p in poi_rotated])-lng_min)/lng_diff
            poi_pos[:, 1] = (torch.tensor([p.y for p in poi_rotated])-lat_min)/lat_diff
        poi_features = ['shop', 'metro_station', 'bus_stop', 'restaurant', 'entertainment', 'park', 'sport', 'school', 'healthcare', 'office']
        if augment:
            random.shuffle(poi_features)
        poi_x = torch.from_numpy(pois[poi_features].values)
        
        nodes = pd.read_csv(base_path + 'nodes.csv')
        if augment:
            road_xs = nodes['x'].values
            road_ys = nodes['y'].values
            road_rotated = gpd.points_from_xy(road_xs, road_ys).rotate(rotation_angle, origin=(lng_center,lat_center))
            road_pos = torch.zeros((nodes.shape[0], 2))
            road_pos[:, 0] = (torch.tensor([p.x for p in road_rotated])-lng_min)/lng_diff
            road_pos[:, 1] = (torch.tensor([p.y for p in road_rotated])-lat_min)/lat_diff
        else:
            road_pos = torch.from_numpy(((nodes[['x','y']]-[lng_min,lat_min])/[lng_diff,lat_diff]).values)
        # road_y = [0,0,0,0,0,0,0,0,0,0] # 10 zeros
        # road_y[int(row['bucket'])] = 1
        road_y = [row['walkability_score'] / 100.0]
        histogram = np.zeros((1 + len(poi_features), histogram_size, histogram_size))

        x_bin_indices = np.digitize(road_pos[:, 0], x_edges) - 1
        y_bin_indices = np.digitize(road_pos[:, 1], y_edges) - 1
        for i in range(road_pos.shape[0]):
            x_bin = x_bin_indices[i]
            y_bin = y_bin_indices[i]
            if 0 <= x_bin < histogram_size and 0 <= y_bin < histogram_size:
                histogram[0, x_bin, y_bin] = 1
        x_bin_indices = np.digitize(poi_pos[:, 0], x_edges) - 1
        y_bin_indices = np.digitize(poi_pos[:, 1], y_edges) - 1
        for i in range(poi_pos.shape[0]):
            x_bin = x_bin_indices[i]
            y_bin = y_bin_indices[i]
            if 0 <= x_bin < histogram_size and 0 <= y_bin < histogram_size:
                histogram[torch.argmax(poi_x[i, :]).item()+1, x_bin, y_bin] = 1
        data = (torch.from_numpy(histogram).to(torch.float32),
                torch.tensor(road_y).to(torch.float32),
                '%s_%d_%d' % (place, int(row['i']), int(row['j'])))
        self.cache[idx] = data
        return data

def custom_collate_fn(batch):
    # Split batch items into first `D` items and the rest
    histogram = default_collate([item[0] for item in batch])
    y = default_collate([item[1] for item in batch])
    idx_list = [item[2] for item in batch]
    return histogram, y, idx_list

batch_size = 32
train_loader = DataLoader(SpatialDatasetSeg(pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/train_summary.csv')),batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(SpatialDatasetSeg(pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/test_summary.csv')),batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

import torch.nn as nn
from transformers import ResNetConfig, ResNetModel


class ResNet(nn.Module):
    def __init__(self, in_channels, output_dim):
        super(ResNet, self).__init__()
        # Load the ResNet model from Hugging Face
        config = ResNetConfig(num_channels=in_channels, layer_type='basic', depths=[2,2], hidden_sizes=[128,256], embedding_size=16)
        self.resnet = ResNetModel(config)
        # Dimension of ResNet output and the extra feature dimension
        resnet_output_dim = config.hidden_sizes[-1]  # This is typically 2048 for ResNet-50
        # combined_dim = resnet_output_dim
        # Define the fully connected layer to produce the desired output dimension
        self.fc = nn.Linear(resnet_output_dim, output_dim)
    def forward(self, image):
        # Pass the image through ResNet to get the pooled output
        resnet_features = self.resnet(image).pooler_output  # shape: (batch_size, resnet_output_dim)
        resnet_features = resnet_features.reshape((resnet_features.shape[0], resnet_features.shape[1]))
        # Concatenate the ResNet output with the extra features
        # combined_features = torch.cat((resnet_features, extra_feature), dim=1)  # shape: (batch_size, combined_dim)
        # Pass through the fully connected layer to get the final output
        output = self.fc(resnet_features)
        return output


model = ResNet(11,1).to(device)


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
    # metric = MulticlassAccuracy(average='micro', num_classes=10)
    num_batches = len(train_loader)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{config['epochs']}"
    )
    iterator = iter(train_loader)
    for batch_idx in progress_bar:
        histogram, y, batch = next(iterator)
        histogram, y  = histogram.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(histogram)
        loss = F.mse_loss(prediction, y)
        # metric.update(torch.argmax(prediction, dim=-1), torch.argmax(y, dim=-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_batches
    # acc = metric.compute()
    # print('train loss: %f, train acc: %s' % ( epoch_loss, str(acc) ))
    print('train loss: %f' % ( epoch_loss) )

def val_step(epoch, best_wmape):
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
    # metric1 = MulticlassAccuracy(average='micro', num_classes=10)
    # metric2 = MulticlassPrecision(average='micro', num_classes=10)
    # metric3 = MulticlassRecall(average='micro', num_classes=10)
    # metric4 = MulticlassF1Score(average='micro', num_classes=10)
    iterator = iter(val_loader)
    for batch_idx in progress_bar:
        histogram, y, batch = next(iterator)
        histogram, y  = histogram.to(device), y.to(device)
        actual = y if actual == None else torch.cat((actual, y))
        batches = batch if batches == None else batches + batch
        with torch.no_grad():
            prediction = model(histogram)
        loss = F.mse_loss(prediction, y)
        # predicted_labels, actual_labels = torch.argmax(prediction, dim=-1), torch.argmax(y, dim=-1)
        # metric1.update(predicted_labels, actual_labels)
        # metric2.update(predicted_labels, actual_labels)
        # metric3.update(predicted_labels, actual_labels)
        # metric4.update(predicted_labels, actual_labels)
        predictions = prediction.cpu() if predictions == None else torch.cat((predictions.cpu(), prediction.cpu()))
        epoch_loss += loss.item()
    # m1 = metric1.compute()
    # m2 = metric2.compute()
    # m3 = metric3.compute()
    # m4 = metric4.compute()
    wmape =   weighted_mean_absolute_percentage_error(predictions.cpu(), actual.cpu())
    epoch_loss = epoch_loss / num_batches
    # print('val loss: %f\t accuracy: %f\t precision: %f\t recall: %f\t f1: %f' % (epoch_loss, m1, m2, m3, m4))
    print('val loss: %f\t wmape: %f' % (epoch_loss, wmape))
    if wmape < best_wmape:
        cur_path = config["output_folder"] + "/walkability_resnet_%.6f.csv" % (best_wmape)
        checkpoint_path = config["output_folder"] + "/walkability_resnet_%.6f.ckpt" % (best_wmape)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        best_wmape = wmape
        # best_wmape = m4
        df = pd.DataFrame()
        df["label"] = batches
        # df["actual"] = torch.argmax(actual, dim=-1).cpu().numpy().flatten()
        df["actual"] = actual.cpu().numpy().flatten()
        df["predicted"] = predictions.cpu().numpy().flatten()
        # df["partition_label"] = labels
        cur_path = config["output_folder"] + "/walkability_resnet_%.6f.csv" % (best_wmape)
        checkpoint_path = config["output_folder"] + "/walkability_resnet_%.6f.ckpt" % (best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)
    return best_wmape


        
best_wmape = 100
best_wmape = 100
best_loss = 1e10
total_time = 0

for epoch in range(1, config['epochs'] + 1):
    start = timer()
    train_step(epoch)
    total_time += timer() - start
    best_wmape = val_step(epoch, best_wmape)
