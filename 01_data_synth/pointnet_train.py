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

    'all_values_parametrized_synth':  ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
           'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
           'k_value_$$',
            'e0', 'e2'],
    'all_values_parametrized_weather':  ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
           'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
           'k_value_$$',
            'e0', 'e2']
}


for label in labeled_outputs:
    print("STARTING FOR %s" % label)
    outputs = labeled_outputs[label]
    is_parametrized = 'parametrized' in label and 'not' not in label
    train_dataset = SpatialDataset(is_train=True,  outputs=outputs, parametrized=is_parametrized, histogram=False, with_noise=False)
    val_dataset = SpatialDataset(is_train=False, outputs=outputs, parametrized=is_parametrized,histogram=False, with_noise=False)
    if 'weather' in label:
        train_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data',is_train=True,  outputs=outputs, parametrized=is_parametrized, histogram=False, with_noise=False, rotate=False)
        val_dataset = SpatialDataset(folder='/rhome/msaee007/bigdata/pointnet_data/p1_real_data',is_train=False, outputs=outputs, parametrized=is_parametrized,histogram=False, with_noise=False, rotate=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    base_parameters['set_abstractions'][0]['mlp'][0] = 2 + train_dataset[0].x.shape[1]
    base_parameters['final_mlp'][-1] = len(outputs)
    
    model = PointNet(**base_parameters).to(device)
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