import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import exists
from torchmetrics.functional import weighted_mean_absolute_percentage_error
from torch_geometric.loader import DataLoader
import pandas as pd
from GeometryEncoder import GeometryEncoder

sys.path.append(os.path.abspath("../01_data_synth/"))

from dataset import SpatialDataset
from tqdm import tqdm

output_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/main_exp_outputs'
device = torch.device('cuda')
max_epochs = 150
batch_size=32

label = 'poly2vec_' + sys.argv[1] # 'synth' or 'weather'

outputs = ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
           'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
           'k_value_$$',
            'e0', 'e2']



def get_all_sequences(arr):
    sequences = []
    start = 0
    current_value = arr[0]
    for i in range(1, len(arr)):
        if arr[i] != current_value:
            sequences.append((current_value, start, i - 1))
            start = i
            current_value = arr[i]
    sequences.append((current_value, start, len(arr) - 1))
    return sequences

import random
def pad_or_random_sample(input1, input2, max_seq_len=1024):
    """
    Pads or randomly samples a list of 2D tensors to fixed length max_seq_len.
    For tensors longer than max_seq_len, selects max_seq_len random timesteps (not contiguous).
    Each tensor must be of shape (seq_len, embedding_dim).
    Returns a tensor of shape (batch_size, max_seq_len, embedding_dim).
    """
    output1 = torch.zeros((max_seq_len, input1[0].shape[-1]), dtype=input1[0].dtype)
    output2 = torch.zeros((max_seq_len, input2[0].shape[-1]), dtype=input2[0].dtype)

    seq_len = len(input1)
    if seq_len > max_seq_len:
        indices = sorted(random.sample(range(seq_len), max_seq_len))
        output1 = input1[indices]
        output2 = input2[indices]
    else:
        output1[0:seq_len] = input1
        output2[0:seq_len] = input2
    return output1, output2


def convert_data(data_loader):
    batches = []
    num_batches = len(data_loader)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Preparing training data"
    )
    iterator = iter(data_loader)
    for batch_idx in progress_bar:
        data = next(iterator)
        pos = data.pos
        feat = data.x
        y = data.y
        indexes = get_all_sequences(data.batch)
        points = torch.zeros((y.shape[0], 1024, 2))
        features = torch.zeros((y.shape[0], 1024, 3))
        for i in range(len(indexes)):
            _, start, end = indexes[i]
            _points = pos[start:end+1, :]
            _features = feat[start:end+1, :]
            _points, _features = pad_or_random_sample(_points, _features)
            points[i, :, :], features[i, :, :] = _points,_features
        batches.append((points,features, y))
    return batches


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

        # Output head
        self.output_head = nn.Sequential(
            nn.Flatten(),  # Flatten the sequence
            nn.Linear(max_seq_len * input_dim, output_dim)
        )

    def forward(self, geometries, lengths, dataset_type, features):
        x = torch.zeros((geometries.shape[0], 1024, 36)).to(device)
        for i in range(geometries.shape[0]):
            x[i, :, :32] = self.geometry_encoder(geometries[i], lengths, dataset_type)
            x[i, :, 32:35] = features[i, :, :]
        # x = self.geometry_encoder(geometries, lengths, dataset_type)
        # x = torch.cat((x,features), axis=1)
        x = self.transformer_encoder(x)
        out = self.output_head(x)
        return out


def train_step(model, optimizer, epoch, train_data):
    """Training Step"""
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_data)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{max_epochs}"
    )
    for batch_idx in progress_bar:
        points, features, y = train_data[batch_idx]
        points = points.to(device)
        features = features.to(device)
        y = y.to(device)
        # data = next(iterator).to(device)
        # pos = data.pos
        # feat = data.x
        # y = data.y.to(device)
        # indexes = get_all_sequences(data.batch)
        # points = torch.zeros((y.shape[0], 1024, 2)).to(device)
        # features = torch.zeros((y.shape[0], 1024, 3)).to(device)
        # for i in range(len(indexes)):
        #     _, start, end = indexes[i]
        #     _points = pos[start:end+1, :]
        #     _features = feat[start:end+1, :]
        #     _points, _features = pad_or_random_sample(_points, _features)
        #     points[i, :, :], features[i, :, :] = _points,_features
        optimizer.zero_grad()
        prediction = model(points.to(device), torch.tensor([]).to(device), "points", features.to(device))
        loss = F.mse_loss(prediction, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_batches
    print('train loss: %f' % ( epoch_loss ))


def val_step(model, epoch, val_data, label, outputs, best_wmape):
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
    for batch_idx in progress_bar:
        points, features, y = val_data[batch_idx]
        points = points.to(device)
        features = features.to(device)
        y = y.to(device)
        # data = next(iterator).to(device)
        # pos = data.pos
        # feat = data.x
        # y = data.y.to(device)
        # indexes = get_all_sequences(data.batch)
        # points = torch.zeros((len(indexes), 1024, 2)).to(device)
        # features = torch.zeros((len(indexes), 1024, 3)).to(device)
        # for i in range(len(indexes)):
        #     _, start, end = indexes[i]
        #     _points = pos[start:end+1]
        #     _features = feat[start:end+1]
        #     points[i, :, :], features[i, :, :] = pad_or_random_sample(_points, _features)
        with torch.no_grad():
            prediction = model(points.to(device), torch.tensor([]).to(device), "points", features.to(device))
        loss = F.mse_loss(prediction, y)
        actual = y if actual == None else torch.cat((actual, y))
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
        for i in range(len(outputs)):
            df[f"actual_{outputs[i]}"] = actual[:, i].cpu().numpy()
            df[f"predicted_{outputs[i]}"] = predictions[:, i].cpu().numpy()
        best_wmape = wmape
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)

    return best_wmape



if __name__ == "__main__":
    is_parametrized = True
    train_dataset = SpatialDataset(is_train=True,  outputs=outputs, parametrized=is_parametrized, histogram=False, with_noise=False)
    val_dataset = SpatialDataset(is_train=False, outputs=outputs, parametrized=is_parametrized,histogram=False, with_noise=False)
    if 'weather' in label:
        train_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data',is_train=True,  outputs=outputs, parametrized=is_parametrized, histogram=False, with_noise=False, rotate=False)
        val_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data',is_train=False, outputs=outputs, parametrized=is_parametrized,histogram=False, with_noise=False, rotate=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_data = convert_data(train_loader)
    val_data = convert_data(val_loader)

    geometry_model = GeometryEncoder(device).to(device)
    model = SimpleTransformerModel(geometry_model)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )

    best_wmape = 10**10
    for i in range(1,max_epochs+1):
            train_step(model, optimizer, i, train_data)
            best_wmape = val_step(model, i, val_data, label, outputs, best_wmape)
