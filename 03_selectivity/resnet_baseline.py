from torch.utils.data import DataLoader
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
from dataset import SpatialDataset
import copy
import torch.nn as nn
from transformers import ResNetConfig, ResNetModel
from torch_geometric.nn import MLP
import time

output_folder = '/rhome/msaee007/bigdata/pointnet_data/selectivity_exp_output'
device = torch.device('cuda')
max_epochs = 150
batch_size=32

# outputs = ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
#            'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
#            'k_value_$$',
#             'e0', 'e2']

# train_dataset = SpatialDataset(is_train=True,  outputs=outputs, parametrized=True, histogram=True, with_noise=False)
# val_dataset = SpatialDataset(is_train=False, outputs=outputs, parametrized=True, histogram=True, with_noise=False)
# train_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data', is_train=True,  outputs=outputs, parametrized=True, histogram=True, with_noise=False, rotate=True)
# val_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data', is_train=False, outputs=outputs, parametrized=True, histogram=True, with_noise=False, rotate=True)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


train_dataset = SpatialDataset(is_train=True, histogram=True)
val_dataset = SpatialDataset(is_train=False,histogram=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)




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
        hists, params, y = next(iterator)
        hists, params, y = hists.to(device), params.to(device), y.to(device)
        hists = hists.reshape(hists.shape[0], 4, hists.shape[-1], hists.shape[-1])
        y = y.reshape((y.shape[0], y.shape[-1]))
        optimizer.zero_grad()
        prediction = model(hists, params)
        loss = F.mse_loss(prediction, y)
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
    total_time = 0
    iterator = iter(val_loader)
    for batch_idx in progress_bar:
        hists, params, y = next(iterator)
        hists, params, y = hists.to(device), params.to(device), y.to(device)
        hists = hists.reshape(hists.shape[0], 4, hists.shape[-1], hists.shape[-1])
        y = y.reshape((y.shape[0], y.shape[-1]))
        with torch.no_grad():
            t1 = time.time()
            prediction = model(hists, params)
            total_time += time.time() - t1
        loss = F.mse_loss(prediction, y)
        actual = y if actual == None else torch.cat((actual, y))
        predictions = prediction if predictions == None else torch.cat((predictions, prediction))
        epoch_loss += loss.item()
    print("total val time: ", total_time)
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
        for i in range(len(outputs)):
            df[f"actual_{outputs[i]}"] = actual[:, i].cpu().numpy()
            df[f"predicted_{outputs[i]}"] = predictions[:, i].cpu().numpy()
        best_wmape = wmape
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)

    return best_wmape


class ResNet(nn.Module):
    def __init__(self, output_dim):
        super(ResNet, self).__init__()
        # Load the ResNet model from Hugging Face
        config = ResNetConfig(num_channels=4, layer_type='basic', depths=[2,2], hidden_sizes=[128,256], embedding_size=16)
        self.resnet = ResNetModel(config)
        # Dimension of ResNet output and the extra feature dimension
        resnet_output_dim = config.hidden_sizes[-1]  # This is typically 2048 for ResNet-50
        combined_dim = resnet_output_dim + 4
        # Define the fully connected layer to produce the desired output dimension
        # self.fc = nn.Linear(combined_dim, output_dim)
        # self.fc = nn.Linear(combined_dim, output_dim)
        self.fc = MLP([combined_dim, 256, 128, 1])
    def forward(self, image, extra_feature):
        # Pass the image through ResNet to get the pooled output
        resnet_features = self.resnet(image).pooler_output  # shape: (batch_size, resnet_output_dim)
        resnet_features = resnet_features.reshape((resnet_features.shape[0], resnet_features.shape[1]))
        # Concatenate the ResNet output with the extra features
        combined_features = torch.cat((resnet_features, extra_feature), dim=1)  # shape: (batch_size, combined_dim)
        # Pass through the fully connected layer to get the final output
        output = self.fc(combined_features)
        return output


model = ResNet(1).to(device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('model trainable parameters: ', pytorch_total_params)

optimizer = torch.optim.Adam(
    model.parameters(), lr=1.0e-4
)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('trainable parameters: ', pytorch_total_params)

best_wmape = 100
for i in range(1,max_epochs+1):
    train_step(model, optimizer, i, train_loader)
    best_wmape = val_step(model, i, val_loader, 'resnet_selectivity', ['selectivity'], best_wmape)

