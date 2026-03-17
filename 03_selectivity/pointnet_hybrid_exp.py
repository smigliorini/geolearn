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

output_folder = '/rhome/msaee007/bigdata/pointnet_data/selectivity_exp_output'
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


def val_step(model, epoch, val_loader, label, best_wmape):
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
    if wmape < best_wmape:
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        df = pd.DataFrame()
        df[f"actual"] = actual[:].cpu().numpy().flatten()
        df[f"predicted"] = predictions[:].cpu().flatten()
        best_wmape = wmape
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)
    return best_wmape

train_dataset = SpatialDataset(is_train=True, histogram=False)
val_dataset = SpatialDataset(is_train=False,histogram=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


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


class SelectivityHybrid(torch.nn.Module):
    def __init__(
        self,
        ratio=1.0, r=2.0, max_neighbors=16, DIM=128
    ):
        super().__init__()
        self.DIM = DIM
        self.sa1  = SetAbstraction(ratio, r, max_neighbors,  [2 + 8, 64, 128, 128])
        config = ResNetConfig(num_channels=128, layer_type='basic', depths=[2,2], hidden_sizes=[128,256], embedding_size=16)
        self.resnet = ResNetModel(config)
        resnet_output_dim = config.hidden_sizes[-1]
        self.fc = MLP([resnet_output_dim, 256, 256, 128, 1])
    def forward(self, data):
        sa0 = (data.x, data.pos, data.batch)
        sa1 = self.sa1(*sa0)
        images = points_to_images(*sa1, self.DIM)
        resnet_features = self.resnet(images).pooler_output
        resnet_features = resnet_features.reshape((resnet_features.shape[0], resnet_features.shape[1]))
        output = self.fc(resnet_features)
        return output


model = SelectivityHybrid().to(device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1.0e-4
)
best_wmape = 100
label = 'hybrid_selectivity'
for i in range(1,max_epochs+1):
        train_step(model, optimizer, i, train_loader)
        best_wmape = val_step(model, i, val_loader, label, best_wmape)
del model
del optimizer
gc.collect()
torch.cuda.empty_cache() 