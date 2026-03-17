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
import geopandas as gpd
from torch_geometric.data import Data, Dataset, Batch
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import random
from collections import defaultdict


output_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/main_exp_outputs'
device = torch.device('cuda')
max_epochs = 150
batch_size=32

label = 'poly2vec_walkability_' 

def custom_stratified_sample(X, y_onehot, sample_size=1024, random_state=42):
    np.random.seed(random_state)
    
    y_labels = np.argmax(y_onehot, axis=1)
    n_samples = X.shape[0]

    if n_samples <= sample_size:
        return X, y_onehot

    # Group indices by class
    class_to_indices = defaultdict(list)
    for idx, cls in enumerate(y_labels):
        class_to_indices[cls].append(idx)

    # First include all singleton classes
    selected_indices = []
    remaining_indices = []
    for cls, indices in class_to_indices.items():
        if len(indices) == 1:
            selected_indices.extend(indices)
        else:
            remaining_indices.append((cls, indices))

    # Calculate remaining slots
    slots_left = sample_size - len(selected_indices)
    if slots_left <= 0:
        raise ValueError("Too many singleton classes to satisfy the sample size constraint.")

    # Total available non-singleton samples
    all_remaining = [idx for _, indices in remaining_indices for idx in indices]
    total_remaining = len(all_remaining)

    # Proportional sampling from remaining classes
    for cls, indices in remaining_indices:
        np.random.shuffle(indices)
        proportion = len(indices) / total_remaining
        n_cls = int(round(proportion * slots_left))
        selected_indices.extend(indices[:n_cls])

    # Trim or pad to exact size (in case of rounding error)
    if len(selected_indices) > sample_size:
        selected_indices = selected_indices[:sample_size]
    elif len(selected_indices) < sample_size:
        # Randomly pad from unused indices
        used = set(selected_indices)
        unused = list(set(range(n_samples)) - used)
        np.random.shuffle(unused)
        selected_indices.extend(unused[:sample_size - len(selected_indices)])

    selected_indices = np.array(selected_indices)
    return X[selected_indices], y_onehot[selected_indices]

class SpatialDataset(Dataset):
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
        lng_min = min(row['east'], row['west']) #min(road_data[:, 4].min(), poi_data[:, 0].min())
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

        # merge the two datasets in one dataframe
        new_pos = torch.zeros(road_pos.shape[0] + poi_pos.shape[0], 2)
        new_x = torch.zeros(road_pos.shape[0] + poi_pos.shape[0], 11)
        new_pos[0:poi_pos.shape[0], :] = poi_pos
        new_pos[poi_pos.shape[0]:, :] = road_pos
        new_x[0:poi_pos.shape[0], :10] = poi_x
        new_x[poi_pos.shape[0]:, 10] = 1

        pos_output = new_pos
        x_output = new_x

        if pos_output.shape[0] > 1024:
            pos_output, x_output = custom_stratified_sample(pos_output.numpy(), x_output.numpy(), sample_size=1024, random_state=42)
            # x_labels = np.argmax(x_output, axis=1)
            # # Stratified sampling
            # pos_output, _, x_output, _ = train_test_split(
            #     pos_output.numpy(), x_output.numpy(),
            #     train_size=1024,
            #     stratify=x_labels,
            #     random_state=0
            # )
            pos_output = torch.from_numpy(pos_output)
            x_output = torch.from_numpy(x_output)
        else:
            _pos_output = torch.zeros((1024, 2))
            _pos_output[:pos_output.shape[0]] = pos_output
            pos_output = _pos_output

            _x_output = torch.zeros((1024, 11))
            _x_output[:x_output.shape[0]] = x_output
            x_output = _x_output
         
        # road_y[int(row['bucket'])] = 1
        # if road_x.shape[0] > 1024:
        #     indices = sorted(random.sample(range(road_pos.shape[0]), 1024))
        #     road_pos = road_pos[indices]
        #     road_x = torch.zeros((road_pos.shape[0], poi_x.shape[1]))
        # else:
        #     _road_pos = torch.zeros((1024, 2))
        #     _road_pos[:road_pos.shape[0]] = road_pos
        #     road_pos = _road_pos
        #     road_x = torch.zeros((road_pos.shape[0], poi_x.shape[1]))
        # if poi_x.shape[0] > 1024:
        #     indices = sorted(random.sample(range(poi_pos.shape[0]), 1024))
        #     poi_pos = poi_pos[indices]
        #     poi_x = poi_x[indices]
        # else:
        #     _poi_pos = torch.zeros((1024, 2))
        #     _poi_x = torch.zeros((1024, poi_x.shape[-1]))
        #     _poi_pos[:poi_pos.shape[0]] = poi_pos
        #     _poi_x[:poi_x.shape[0]] = poi_x
        #     poi_pos = _poi_pos
        #     poi_x = _poi_x
        # data = road_pos.float(), road_x.float(), poi_pos.float(), poi_x.float(), torch.tensor(road_y).float(), '%s_%d_%d' % (place, int(row['i']), int(row['j']))
        data = pos_output.float(), x_output.float(), torch.tensor(road_y).float(), '%s_%d_%d' % (place, int(row['i']), int(row['j']))
        self.cache[idx] = data
        return data


# class CrossAttentionBlock(nn.Module):
#     def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
#         super().__init__()
        
#         # Cross-attention layers using nn.MultiheadAttention
#         self.cross_attention_1to2 = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
#         self.cross_attention_2to1 = nn.MultiheadAttention(
#             embed_dim=d_model,
#             num_heads=num_heads,
#             dropout=dropout,
#             batch_first=True
#         )
        
#         # Layer normalization
#         self.norm1_1 = nn.LayerNorm(d_model)
#         self.norm1_2 = nn.LayerNorm(d_model)
#         self.norm2_1 = nn.LayerNorm(d_model)
#         self.norm2_2 = nn.LayerNorm(d_model)
        
#         # Feed-forward networks
#         self.ffn1 = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model),
#             nn.Dropout(dropout)
#         )
        
#         self.ffn2 = nn.Sequential(
#             nn.Linear(d_model, d_ff),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_ff, d_model),
#             nn.Dropout(dropout)
#         )
        
#         self.dropout = nn.Dropout(dropout)
    
#     def forward(self, sent1, sent2, sent1_mask=None, sent2_mask=None):
#         # Cross-attention: sent1 attends to sent2
#         attn_output_1, attn_weights_1 = self.cross_attention_1to2(
#             query=sent1,
#             key=sent2,
#             value=sent2,
#             key_padding_mask=sent2_mask,
#             need_weights=True
#         )
#         sent1 = self.norm1_1(sent1 + self.dropout(attn_output_1))
        
#         # Cross-attention: sent2 attends to sent1
#         attn_output_2, attn_weights_2 = self.cross_attention_2to1(
#             query=sent2,
#             key=sent1,
#             value=sent1,
#             key_padding_mask=sent1_mask,
#             need_weights=True
#         )
#         sent2 = self.norm1_2(sent2 + self.dropout(attn_output_2))
        
#         # Feed-forward networks
#         sent1 = self.norm2_1(sent1 + self.ffn1(sent1))
#         sent2 = self.norm2_2(sent2 + self.ffn2(sent2))
        
#         return sent1, sent2, attn_weights_1, attn_weights_2

# class TwoSentenceCrossAttentionModel(nn.Module):
#     def __init__(self, geometry_model, d_model=44, num_heads=4, num_layers=4, 
#                  d_ff=2048, max_len=1024, dropout=0.1, num_classes=3):
#         super().__init__()
        
#         self.d_model = d_model
#         self.geometry_encoder = geometry_model
        
#         # Cross-attention layers
#         self.cross_attention_layers = nn.ModuleList([
#             CrossAttentionBlock(d_model, num_heads, d_ff, dropout)
#             for _ in range(num_layers)
#         ])
        
#         #
#         self.mlp = nn.Sequential(
#             nn.Linear(d_model * 4, d_model),  # Concatenated representations
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(d_model, 1)
#         )
#         self.dropout = nn.Dropout(dropout)
        
#         # Initialize weights
#         self._init_weights()
    
#     def _init_weights(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
        
#     def forward(self, road_pos, road_x, poi_pos, poi_x):
        
#         # Create embeddings
#         roads = torch.zeros((road_pos.shape[0], 1024, 44)).to(device)
#         pois = torch.zeros((poi_pos.shape[0], 1024, 44)).to(device)
#         for i in range(roads.shape[0]):
#             roads[i, :, :32] = self.geometry_encoder(road_pos[i], torch.tensor([]).to(device), "points")
#             pois[i, :, :32] = self.geometry_encoder(poi_pos[i], torch.tensor([]).to(device), "points")
#             pois[i, :, 32:42] = poi_x[i, :, :]
        
#         roads_mask = (road_pos == 0).all(dim=2)
#         pois_mask = (poi_pos == 0).all(dim=2)
#         roads = self.dropout(roads)
#         pois = self.dropout(pois)
        
#         # Apply cross-attention layers
#         all_attention_weights = []
#         for layer in self.cross_attention_layers:
#             roads_emb, pois_emb, attn_w1, attn_w2 = layer(
#                 roads, pois, roads_mask, pois_mask
#             )
#             all_attention_weights.append((attn_w1, attn_w2))
        
#         # Pool representations (mean pooling, ignoring padding)
#         roads_lens = (~roads_mask).sum(dim=1, keepdim=True).float()
#         pois_lens = (~pois_mask).sum(dim=1, keepdim=True).float()
        
#         roads_pooled = roads_emb.sum(dim=1) / roads_lens
#         pois_pooled = pois_emb.sum(dim=1) / pois_lens
        
#         # Create combined representation
#         diff = roads_pooled - pois_pooled
#         prod = roads_pooled * pois_pooled
#         combined = torch.cat([roads_pooled, pois_pooled, diff, prod], dim=1)        
#         return self.mlp(combined)


class SimpleTransformerModel(nn.Module):
    def __init__(self, geometry_model, input_dim=44, output_dim=1, max_seq_len=1024, nhead=4, num_layers=4, dim_feedforward=128):
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
        x = torch.zeros((geometries.shape[0], 1024, 44)).to(device)
        for i in range(geometries.shape[0]):
            x[i, :, :32] = self.geometry_encoder(geometries[i], lengths, dataset_type)
            x[i, :, 32:32+features.shape[-1]] = features[i, :, :]
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
    lengths = torch.full((batch_size, 1024, 5), 5).to(device)
    progress_bar = tqdm(
        range(num_batches),
        desc=f"Training Epoch {epoch}/{max_epochs}"
    )
    iterator = iter(train_data)
    for batch_idx in progress_bar:
        # road_pos, road_x, poi_pos, poi_x, y, labels = next(iterator)
        # road_pos, road_x, poi_pos, poi_x, y = road_pos.to(device), road_x.to(device), poi_pos.to(device), poi_x.to(device), y.to(device)
        pos, x, y, _labels = next(iterator)
        pos, x, y = pos.to(device), x.to(device), y.to(device)
        optimizer.zero_grad()
        prediction = model(pos, torch.tensor([]).to(device), "points", x)
        loss = F.mse_loss(prediction, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / num_batches
    print('train loss: %f' % ( epoch_loss ))


def val_step(model, epoch, val_data, label, best_wmape):
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
    labels = []
    iterator = iter(val_data)
    for batch_idx in progress_bar:
        pos, x, y, _labels = next(iterator)
        actual = y if actual == None else torch.cat((actual, y))
        pos, x, y = pos.to(device), x.to(device), y.to(device)
        labels += _labels
        with torch.no_grad():
            prediction = model(pos, torch.tensor([]).to(device), "points", x)
        loss = F.mse_loss(prediction, y)
        predictions = prediction.cpu() if predictions == None else torch.cat((predictions, prediction.cpu()))
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / num_batches
    wmape = weighted_mean_absolute_percentage_error(predictions.cpu(), actual.cpu())
    print('val loss: %f\t wmape: %f' % (epoch_loss, wmape))
    if wmape < best_wmape:
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        if exists(cur_path):
            os.remove(cur_path)
        if exists(checkpoint_path):
            os.remove(checkpoint_path)
        df = pd.DataFrame()
        df["actual"] = actual.cpu().numpy().flatten()
        df["predicted"] = predictions.cpu().numpy().flatten()
        df["partition_label"] = labels
        best_wmape = wmape
        cur_path = output_folder + "/%s_%.6f.csv" % (label, best_wmape)
        checkpoint_path = output_folder + "/%s_%.6f.ckpt" % (label, best_wmape)
        df.to_csv(cur_path, index=False)
        torch.save(model.state_dict(), checkpoint_path)

    return best_wmape


def custom_collate_fn(batch):
    # batch is a list of tuples: [(tensor1, tensor2, string), ...]
    tensor1_batch = torch.stack([item[0] for item in batch])
    tensor2_batch = torch.stack([item[1] for item in batch])
    tensor3_batch = torch.stack([item[2] for item in batch])
    # tensor4_batch = torch.stack([item[3] for item in batch])
    # tensor5_batch = torch.stack([item[4] for item in batch])
    string_batch = [item[3] for item in batch]  # keep as list
    # return tensor1_batch, tensor2_batch, tensor3_batch, tensor4_batch, tensor5_batch, string_batch
    return tensor1_batch, tensor2_batch, tensor3_batch, string_batch

if __name__ == "__main__":
    is_parametrized = True
    train_dataset = SpatialDataset(pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/train_summary.csv'))
    # train_dataset = SpatialDataset(is_train=True, histogram=False)
    # val_dataset = SpatialDataset(is_train=False,histogram=False)
    val_dataset = SpatialDataset(pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/walkability_data/test_summary.csv'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # train_data = convert_data(train_loader, collate_fn=custom_collate_fn)
    # val_data = convert_data(val_loader, collate_fn=custom_collate_fn)

    geometry_model = GeometryEncoder(device).to(device)
    model = SimpleTransformerModel(geometry_model)
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )

    best_wmape = 10**10
    for i in range(1,max_epochs+1):
            train_step(model, optimizer, i, train_loader)
            best_wmape = val_step(model, i, val_loader, label, best_wmape)
