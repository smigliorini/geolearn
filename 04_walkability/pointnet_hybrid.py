from pathlib import Path
from typing import Optional
import torch
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional import multiclass_accuracy, multiclass_confusion_matrix, multiclass_f1_score, multiclass_precision, multiclass_recall
import torch_scatter
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
from torch_geometric.nn import MLP, fps, global_max_pool, knn_interpolate
from torch_geometric.nn import PointNetConv
#from torchvision import transforms
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import json
import os
from os.path import exists
import geopandas as gpd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
from torch_cluster import radius

import torch.nn as nn
from transformers import ResNetConfig, ResNetModel



# def radius(
#     x: torch.Tensor,
#     y: torch.Tensor,
#     r: float,
#     batch_x: Optional[torch.Tensor] = None,
#     batch_y: Optional[torch.Tensor] = None,
#     max_num_neighbors: int = 32,
#     num_workers: int = 1,
#     batch_size: Optional[int] = None,
# ) -> torch.Tensor:
#     if x.numel() == 0 or y.numel() == 0:
#         return torch.empty(2, 0, dtype=torch.long, device=x.device)

#     x = x.view(-1, 1) if x.dim() == 1 else x
#     y = y.view(-1, 1) if y.dim() == 1 else y
#     x, y = x.contiguous(), y.contiguous()

#     if batch_size is None:
#         batch_size = 1
#         if batch_x is not None:
#             assert x.size(0) == batch_x.numel()
#             batch_size = int(batch_x.max()) + 1
#         if batch_y is not None:
#             assert y.size(0) == batch_y.numel()
#             batch_size = max(batch_size, int(batch_y.max()) + 1)
#     assert batch_size > 0

#     ptr_x: Optional[torch.Tensor] = None
#     ptr_y: Optional[torch.Tensor] = None

#     if batch_size > 1:
#         assert batch_x is not None
#         assert batch_y is not None
#         arange = torch.arange(batch_size + 1, device=x.device)
#         ptr_x = torch.bucketize(arange, batch_x)
#         ptr_y = torch.bucketize(arange, batch_y)

#     return torch.ops.torch_cluster.radius(x, y, ptr_x, ptr_y, r,
#                                           max_num_neighbors, num_workers)



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
        # poi_y = None
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
        road_x = torch.zeros((road_pos.shape[0], poi_x.shape[1]))
        # road_y = [0,0,0,0,0,0,0,0,0,0] # 10 zeros
        road_y = [row['walkability_score'] / 100.0]
        # road_y[int(row['bucket'])] = 1
        road_data = Data(pos=road_pos.to(torch.float32),
                    x=road_x.to(torch.float32),
                    y=torch.tensor(road_y).to(torch.float32))
        poi_data = Data(pos=poi_pos.to(torch.float32),
                    x=poi_x.to(torch.float32),
                    y=None
                    )
        data = road_data, poi_data, '%s_%d_%d' % (place, int(row['i']), int(row['j']))
        self.cache[idx] = data
        return data

def points_to_images(x, pos, batch, N):
    # pos must be normalized to [0,1] in both dimensions
    rows = (pos[:, 1] * N).long().clamp(0, N - 1)
    cols = (pos[:, 0] * N).long().clamp(0, N - 1)
    index = (batch * N * N) + rows * N + cols
    batch_size = 1
    if batch is not None:
        assert pos.size(0) == batch.numel()
        batch_size = int(batch.max()) + 1
    assert batch_size > 0
    pixel_features = torch_scatter.scatter(x, index, dim = 0, dim_size=batch_size*N*N, reduce = 'max')
    images = pixel_features.reshape((batch_size,N,N,x.shape[-1])).permute((0, 3, 1, 2))
    return images

def images_to_points(images, pos, batch):
    # pos must be normalized to [0,1] in both dimensions
    N = images.shape[-1]
    rows = (pos[:, 1] * N).long().clamp(0, N - 1)
    cols = (pos[:, 0] * N).long().clamp(0, N - 1)
    return pos, images[batch, :, rows, cols]

class SetAbstraction(torch.nn.Module):
    def __init__(self, ratio, r, max_neighbors, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.max_neighbors = max_neighbors
        self.conv = PointNetConv(nn, add_self_loops=True)
    def forward(self, x, pos, batch):
        idx = list(range(pos.shape[0])) if self.ratio == 1.0 else fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                        max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class SetAbstractionPair(torch.nn.Module):
    def __init__(self, r, max_neighbors, nn):
        super().__init__()
        self.r = r
        self.max_neighbors = max_neighbors
        self.conv = PointNetConv(nn, add_self_loops=True)
    def forward(self, x1, pos1, batch1, x2, pos2, batch2):
        # idx = list(range(pos.shape[0])) if self.ratio == 1.0 else fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos2, pos1, self.r, batch2, batch1,
                        max_num_neighbors=self.max_neighbors)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv((x2, x1), (pos2, pos1), edge_index)
        return x, pos1, batch1

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



class PointNetRoadPOI(torch.nn.Module):
    def __init__(
        self,
        ratio=1.0, r=2.0, max_neighbors=16, DIM=128
    ):
        super().__init__()
        self.DIM = DIM
        # Input channels account for both `pos` and node features.
        # self.sa1_road_from_poi = SetAbstractionPair( # get features for pois from roads
        #     2.0,
        #     16,
        #     MLP([12, 32, 64, 64, 128]) # 8 = (2 dimensions for the points) + (6 dimensions for the features)
        # ) # roads are the references sets, and we compute features for pois
        # self.sa1_poi_from_road = SetAbstractionPair( # get features for pois from roads
        #     2.0,
        #     16,
        #     MLP([12, 32, 64, 64, 128]) # 8 = (2 dimensions for the points) + (6 dimensions for the features)
        # ) # roads are the references sets, and we compute features for pois
        # num_channels is the same is the feature dimension from SA1
        self.sa1_poi  = SetAbstraction(ratio, r, max_neighbors,  MLP([2 + 10, 64, 128, 128]))
        self.sa1_road = SetAbstraction(ratio, r, max_neighbors, MLP([2 + 0 , 64, 128, 128]))
        config = ResNetConfig(num_channels=256, layer_type='basic', depths=[2,2], hidden_sizes=[256,512], embedding_size=16)
        self.resnet = ResNetModel(config)
        resnet_output_dim = config.hidden_sizes[-1]
        self.fc = MLP([resnet_output_dim, 512, 256, 256, 128, 1])
    def forward(self, road_data, poi_data):
        sa0_road = (road_data.x, road_data.pos, road_data.batch)
        sa0_poi = (poi_data.x, poi_data.pos, poi_data.batch)
        sa1_poi = self.sa1_poi(*sa0_poi)
        poi_images = points_to_images(*sa1_poi, self.DIM)
        sa1_road = self.sa1_road(None, sa0_road[1], sa0_road[2])
        road_images = points_to_images(*sa1_road, self.DIM)
        stacked_images = torch.cat([poi_images, road_images], dim=1)
        resnet_features = self.resnet(stacked_images).pooler_output
        resnet_features = resnet_features.reshape((resnet_features.shape[0], resnet_features.shape[1]))
        output = self.fc(resnet_features)
        return output


with open('./config_general.json') as f:
    config = json.load(f)


label = 'general' #sys.argv[1]
with open('./config_%s.json' % label) as f:
    config_exp = json.load(f)

device = (torch.device(config['device']))

#batch_size = int(sys.argv[2])

# train_loader = DataLoader(SpatialDatasetSeg(), batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(SpatialDatasetSeg(), batch_size=batch_size, shuffle=True)

batch_size = 32
# max_nodes = 192000
# train_dataset = SpatialDatasetSeg(pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/train_summary.csv'))

# # node_sizes = {}
# # for i in range(len(train_dataset)):
# #     node_sizes[i] = train_dataset[i][0].pos.shape[0]
# # import binpacking
# # batch_indexes = binpacking.to_constant_volume(node_sizes,max_nodes)
# batch_indexes = np.arange(len(train_dataset))
# np.random.shuffle(batch_indexes)
# r = len(batch_indexes) % batch_size
# remainder = batch_indexes[len(batch_indexes)-r:].tolist()
# batch_indexes = batch_indexes[:len(batch_indexes)-r].reshape(((len(batch_indexes)-r)//batch_size,batch_size)).tolist()
# if len(remainder):
#     batch_indexes.append(remainder)

# train_batches = []
# print('Preparing training data...')
# for batch_idx in tqdm(range(len(batch_indexes))):
#     sample_indexes = batch_indexes[batch_idx]
#     batch_road_data = []
#     batch_poi_data = []
#     for i in sample_indexes:
#         road_data, poi_data, _ = train_dataset[i]
#         batch_road_data.append(road_data)
#         batch_poi_data.append(poi_data)
#     # batch_data = [train_dataset[i] for i in sample_indexes]
#     road_data = Batch.from_data_list(batch_road_data)
#     poi_data = Batch.from_data_list(batch_poi_data)
#     train_batches.append((road_data, poi_data))

# val_dataset = SpatialDatasetSeg(pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/test_summary.csv'))
# batch_indexes2 = np.arange(len(val_dataset))
# np.random.shuffle(batch_indexes2)
# r = len(batch_indexes2) % batch_size
# remainder = batch_indexes2[len(batch_indexes2)-r:].tolist()
# batch_indexes2 = batch_indexes2[:len(batch_indexes2)-r].reshape(((len(batch_indexes2)-r)//batch_size,batch_size)).tolist()
# if len(remainder):
#     batch_indexes2.append(remainder)

# val_batches = []
# print('Preparing validation data...')
# for batch_idx in tqdm(range(len(batch_indexes2))):
#     sample_indexes = batch_indexes2[batch_idx]
#     batch_road_data = []
#     batch_poi_data = []
#     labels = []
#     for i in sample_indexes:
#         road_data, poi_data, label = val_dataset[i]
#         batch_road_data.append(road_data)
#         batch_poi_data.append(poi_data)
#         labels.append(label)
#     # batch_data = [train_dataset[i] for i in sample_indexes]
#     road_data = Batch.from_data_list(batch_road_data)
#     poi_data = Batch.from_data_list(batch_poi_data)
#     val_batches.append((road_data, poi_data, labels))


import pickle
import gzip

def save_dict_compressed(dictionary, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict_compressed(filename):
    with gzip.open(filename, 'rb') as f:
        return pickle.load(f)

# save_dict_compressed(train_batches, '/rhome/msaee007/bigdata/pointnet_data/train_batches.pickle.gzip')
train_batches = load_dict_compressed('/rhome/msaee007/bigdata/pointnet_data/train_batches.pickle.gzip')
# save_dict_compressed(val_batches, '/rhome/msaee007/bigdata/pointnet_data/val_batches.pickle.gzip')
val_batches = load_dict_compressed('/rhome/msaee007/bigdata/pointnet_data/val_batches.pickle.gzip')

# node_sizes = {}
# for i in range(len(val_dataset)):
#     node_sizes[i] = val_dataset[i][0].pos.shape[0]
# import binpacking
# batch_indexes2 = binpacking.to_constant_volume(node_sizes,max_nodes)


model = PointNetRoadPOI(
    # config['set_abstraction_ratio_1'],
    # config['set_abstraction_ratio_2'],
    # config['set_abstraction_radius_1'],
    # config['set_abstraction_radius_2'],
    # config['max_neighbors'],
    # config['dropout'],
    # 1,
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
    num_batches = len(train_batches)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{config['epochs']}"
    )
    for batch_idx in progress_bar:
        road_data, poi_data = train_batches[batch_idx]
        road_data, poi_data = road_data.to(device), poi_data.to(device)
        optimizer.zero_grad()
        prediction = model(road_data, poi_data)
        # print(prediction)
        # print(prediction.shape, road_data.y.shape)
        # loss = F.cross_entropy(prediction, road_data.y.reshape(prediction.shape))
        loss = F.mse_loss(prediction, road_data.y.reshape(prediction.shape))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_batches
    print('train loss: %f' % ( epoch_loss ))

def val_step(epoch, best_wmape):
    """Validation Step"""
    model.eval()
    epoch_loss = 0.0
    # num_batches = len(val_loader)
    num_batches = len(val_batches)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Validation Epoch {epoch}/{config['epochs']}"
    )
    actual = None
    predictions = None
    labels = []
    # metrics = {
    #     'accuracy': multiclass_accuracy, 
    #     'confusion_matrix': multiclass_confusion_matrix, 
    #     'f1': multiclass_f1_score, 
    #     'precision': multiclass_precision, 
    #     'recall': multiclass_recall
    # }

    # metrics_values = {}

    for batch_idx in progress_bar:
        road_data, poi_data, _labels = val_batches[batch_idx]
        road_data, poi_data = road_data.to(device), poi_data.to(device)
        labels += _labels
        actual = road_data.y.cpu() if actual == None else torch.cat((actual, road_data.y.cpu()))
        with torch.no_grad():
            prediction = model(road_data, poi_data)
        # loss = F.cross_entropy(prediction, road_data.y.reshape(prediction.shape))
        loss = F.mse_loss(prediction, road_data.y.reshape(prediction.shape))
        # predictions += prediction.cpu()
        # _prediction = None
        # for i in range(len(sample_indexes)):
        #     predictions.append()
        #     _mean = mean[2] #batch_mean[i]
        #     _std = std[2] #batch_std[i]
        #     __prediction = (prediction[road_data.batch == i].cpu() * label_std) + label_mean
        #     _prediction = __prediction.cpu() if _prediction == None else torch.cat((_prediction.cpu(), __prediction.cpu()))
        predictions = prediction.cpu() if predictions == None else torch.cat((predictions, prediction.cpu()))
        epoch_loss += loss.item()
    # print(actual.shape, predictions.shape)
    epoch_loss = epoch_loss / num_batches
    # print('val loss: %f' % (epoch_loss))
    # actual = torch.argmax(actual.reshape(predictions.shape), dim=-1)
    # predictions = torch.argmax(predictions, dim=-1)
    # for m in metrics:
    #     metrics_values[m] = metrics[m](predictions, actual, num_classes=10)
    #     print(m, metrics_values[m])
    wmape = weighted_mean_absolute_percentage_error(predictions.flatten(), actual.flatten())
    print('val loss: %f\t wmape: %f' % (epoch_loss, wmape))
    # if metrics_values['f1'] > best_f1:
    if wmape < best_wmape:
        cur_path = config["output_folder"] + "/walkability_hybrid_%.6f.csv" % (best_wmape)
        checkpoint_path = config["output_folder"] + "/walkability_hybrid_%.6f.ckpt" % (best_wmape)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        best_wmape = wmape
        # wmape = float(metrics_values['f1'].item())
        df = pd.DataFrame()
        df["actual"] = actual.cpu().numpy().flatten()
        df["predicted"] = predictions.cpu().numpy().flatten()
        df["partition_label"] = labels
        cur_path = config["output_folder"] + "/walkability_hybrid_%.6f.csv" % (best_wmape)
        checkpoint_path = config["output_folder"] + "/walkability_hybrid_%.6f.ckpt" % (best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)
    return wmape


        
# print("SAMPLE_SIZE: %d, BATCH_SIZE: %d" % (sample_size, batch_size))

best_wmape = 100
best_loss = 1e10
total_time = 0

for epoch in range(1, config['epochs'] + 1):
    start = timer()
    train_step(epoch)
    total_time += timer() - start
    # if epoch % 10 == 0:
    start = timer()
    best_wmape = val_step(epoch, best_wmape)
    end = timer()
    print('Validation time: %s' % str(end-start))
