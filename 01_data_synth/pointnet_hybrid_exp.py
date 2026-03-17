from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torchmetrics.functional import weighted_mean_absolute_percentage_error
import os
from os.path import exists
import gc 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import json
from pointnet import PointNet
from dataset import SpatialDataset
import copy

output_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/main_exp_outputs'
device = torch.device('cuda')
max_epochs = 150
batch_size=32


def train_step(model, optimizer, epoch, train_loader):
    """Training Step"""
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{max_epochs}"
    )
    iterator = iter(train_loader)
    for batch_idx in progress_bar:
        data = next(iterator).to(device)
        optimizer.zero_grad()
        prediction = model(data)
        loss = F.mse_loss(prediction, data.y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_batches
    print('train loss: %f' % ( epoch_loss ))


def val_step(model, epoch, val_loader, label, outputs, best_wmape):
    """Validation Step"""
    model.eval()
    epoch_loss = 0.0
    num_batches = len(val_loader)

    progress_bar = tqdm(
        range(num_batches),
        desc=f"Validation Epoch {epoch}/{max_epochs}"
    )
    actual = None
    predictions = None
    iterator = iter(val_loader)
    for batch_idx in progress_bar:
        data = next(iterator).to(device)
        with torch.no_grad():
            prediction = model(data)
        loss = F.mse_loss(prediction, data.y)
        actual = data.y if actual == None else torch.cat((actual, data.y))
        predictions = prediction if predictions == None else torch.cat((predictions, prediction))
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / num_batches
    wmape = weighted_mean_absolute_percentage_error(predictions, actual)
    print('val loss: %f\t wmape: %f' % (epoch_loss, wmape))
    if epoch_loss < best_wmape:
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        df = pd.DataFrame()
        for i in range(len(outputs)):
            df[f"actual_{outputs[i]}"] = actual[:, i].cpu().numpy()
            df[f"predicted_{outputs[i]}"] = predictions[:, i].cpu().numpy()
        best_wmape = epoch_loss
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)

    return best_wmape


base_parameters = {
    'set_abstractions': [
            {'ratio': 0.75, 'radius': 0.1, 'max_neighbors': 16, 'mlp': [2 + 1, 32, 32, 64]},
            {'ratio': 0.5, 'radius': 0.1, 'max_neighbors': 16, 'mlp': [2 + 64, 128, 128, 256]}
    ],
    'global_abstraction': {'mlp': [2 + 256, 512, 512, 1024]},
    'final_mlp': [1024, 512, 512, 128, 9],
    'dropout':0.1
}


labeled_outputs = {
    # 'hotpots_parametrized': ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
    #        'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y'],

    # 'hotspots_64_only': ['hotspots_64_min_val', 'hotspots_64_min_x', 'hotspots_64_min_y',
    #        'hotspots_64_max_val', 'hotspots_64_max_x', 'hotspots_64_max_y'],
    
    # 'k_value_0.25_only': ['k_value_0.25'],

    # 'k_value_parametrized': ['k_value_$$'],

    # 'box_counts_only': ['e0', 'e2'],

    # 'all_values_not_parameterized':  ['hotspots_64_min_val', 'hotspots_64_min_x', 'hotspots_64_min_y',
    #        'hotspots_64_max_val', 'hotspots_64_max_x', 'hotspots_64_max_y',
    #        'k_value_0.25',
    #         'e0', 'e2'],

    'synopsis_hybrid_parametrized_synth':  ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
           'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
           'k_value_$$',
            'e0', 'e2']
}

from pointnet import SetAbstraction
import torch_scatter
from torch_geometric.nn import MLP
from transformers import ResNetConfig, ResNetModel

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


label = 'synopsis_hybrid_parametrized_synth'
print("STARTING FOR %s" % label)
outputs = labeled_outputs[label]
is_parametrized = 'parametrized' in label and 'not' not in label
train_dataset = SpatialDataset(is_train=True,  outputs=outputs, parametrized=is_parametrized, histogram=False, with_noise=False)
val_dataset = SpatialDataset(is_train=False, outputs=outputs, parametrized=is_parametrized,histogram=False, with_noise=False)
# train_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data',is_train=True,  outputs=outputs, parametrized=is_parametrized, histogram=False, with_noise=False, rotate=False)
# val_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data',is_train=False, outputs=outputs, parametrized=is_parametrized,histogram=False, with_noise=False, rotate=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


import torch
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn import PointNetConv
from torch_geometric.nn import MLP, fps, global_max_pool, knn_interpolate

class PointsToImage(torch.nn.Module):
    def __init__(self, radius, max_neighbors, dim, mlp):
        super().__init__()
        self.r = radius
        self.max_neighbors = max_neighbors
        self.dim = dim
        self.conv = PointNetConv(MLP(mlp), add_self_loops=True)
    def forward(self, x, pos, batch):
        idx = list(range(pos.shape[0]))
        edge_row, edge_col = radius(pos, pos[idx], self.r, batch, batch[idx],
                        max_num_neighbors=self.max_neighbors)
        # convert edge_row to pixel_index
        batch_size = 1
        if batch is not None:
            assert pos.size(0) == batch.numel()
            batch_size = int(batch.max()) + 1
        assert batch_size > 0
        N = self.dim
        rows = (pos[:, 1] * N).long().clamp(0, N - 1)
        cols = (pos[:, 0] * N).long().clamp(0, N - 1)
        index = (batch * N * N) + rows * N + cols
        edge_row = index[edge_row]
        # here we convrted the original target index (edge_row) to the pixel id the point belongs to
        # this way all points that belong to the same pxiel gets aggregated together
        edge_index = torch.stack([edge_col, edge_row], dim=0)
        x_dst = None if x is None else x[idx]
        pixel_features = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        images = pixel_features.reshape((batch_size,N,N,x.shape[-1])).permute((0, 3, 1, 2))
        return images

class ImageToPoints(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn
    def forward(self, images, x, pos, batch):
        N = images.shape[-1]
        B, C, H, W = images.shape
        pixels_x = images.permute(0, 2, 3, 1).reshape(B * H * W, C)
        D = H # assume image is square
        pixel_centers = torch.linspace(1/(D*2), 1-1/(D*2), D, device=images.device)
        pos_x = pixel_centers.repeat(D).reshape((D,D))
        pos_y = pos_x.T
        pixels_pos = torch.stack((pos_x.flatten(), pos_y.flatten()), dim=-1).repeat(B)
        pixels_batch = torch.arange(B).repeat_interleave(D*D)
        interpolated_x = knn_interpolate(pixels_x, pixels_pos, pos, pixels_batch, batch, k=self.k)
        if x is not None:
            x = torch.cat([interpolated_x, x], dim=1)
        x = self.nn(x)
        return x, pos, batch


class SynopsisHybrid(torch.nn.Module):
    def __init__(
        self,
        ratio=1.0, r=2.0, max_neighbors=16, DIM=128
    ):
        super().__init__()
        self.DIM = DIM
        self.sa1  = SetAbstraction(1.0, .1, 16,  [2 + train_dataset[0].x.shape[-1], 32, 64, 128])
        config = ResNetConfig(num_channels=128, layer_type='basic', depths=[2,2], hidden_sizes=[256,512], embedding_size=16)
        self.resnet = ResNetModel(config)
        resnet_output_dim = config.hidden_sizes[-1]
        self.fc = MLP([resnet_output_dim, 256, 256, 128, len(outputs)])
    def forward(self, data):
        sa0 = (data.x, data.pos, data.batch)
        sa1 = self.sa1(*sa0)
        images = points_to_images(*sa1, self.DIM)
        resnet_features = self.resnet(images).pooler_output
        resnet_features = resnet_features.reshape((resnet_features.shape[0], resnet_features.shape[1]))
        output = self.fc(resnet_features)
        return output
model = SynopsisHybrid().to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1.0e-4
)
best_wmape = 10**10
for i in range(1,max_epochs+1):
        train_step(model, optimizer, i, train_loader)
        best_wmape = val_step(model, i, val_loader, label, outputs, best_wmape)
del model
del optimizer
gc.collect()
torch.cuda.empty_cache() 