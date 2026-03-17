import pandas as pd
from pathlib import Path
import json
import time
from sklearn.neighbors import KDTree
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader
from dataset import SpatialDataset, get_histogram
from pointnet import PointNet
import torch
from transformers import ResNetConfig, ResNetModel
import torch.nn as nn
from torch_geometric.data import Data
from itertools import product
import binpacking
from torch_geometric.data import Batch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

# Function to measure memory usage (obtained using ChatGPT)
def measure_memory_usage(model, input_tensor, mode="train"):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    pred_time = 0.0
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    if mode == "train":
        model.train()
        torch.cuda.synchronize()
    else:
        model.eval()
        torch.cuda.synchronize()
    # Run forward pass
    with torch.set_grad_enabled(mode == "train"):
        torch.cuda.synchronize()
        start_event.record()
        output = model(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        pred_time = start_event.elapsed_time(end_event)
    # Get max memory allocated
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
    return max_memory, pred_time



def read_labels(path, dist, selected_samples):
    with open(path, 'r') as file:
        lines = file.readlines()
        records = {}
        for l in lines:
            j = json.loads(l)
            id = j['dataset_id']
            id = id[id.rfind('/')+1:]
            if '.csv' in id:
                id = id[:id.rfind('.csv')]
            if not('_' in id):
                id = int(id)
            is_selected = True
            if type(selected_samples) != type(None):
                is_selected = not selected_samples[(selected_samples['id'] == id) & (selected_samples['dist'] == dist)].empty
            if is_selected:
                records[id] = j
        return records

base_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data'
val_samples = pd.read_csv(base_folder + '/val_samples.csv')
dists = ['uniform', 'diagonal', 'gaussian', 'sierpinski']
labels_files = {d: str(Path(f'{base_folder}/{d}/data_summary.csv')) for d in dists}
labels_files['weather'] = '/rhome/msaee007/bigdata/pointnet_data/p1_real_data/weather/data_summary.csv'
val_samples2 = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/p1_real_data/val_samples.csv')

labels_files['bit'] = '//rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/bit/data_summary.csv'
labels_files['parcel'] = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/parcel/data_summary.csv'
labels_files['uniform_large'] = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/uniform/data_summary.csv'
labels_files['gaussian_large'] = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/gaussian/data_summary.csv'
labels_files['diagonal_large'] = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/diagonal/data_summary.csv'
labels_files['sierpinski_large'] = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/sierpinski/data_summary.csv'
labels = {}

for d in labels_files:
    if d in dists:
        labels[d] = read_labels(labels_files[d], d, val_samples)
    elif d == 'weather':
        labels[d] = read_labels(labels_files[d], d, val_samples2)
    else:
        labels[d] = read_labels(labels_files[d], d, None)


outputs = ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
           'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
           'k_value_$$',
            'e0', 'e2']
with open(base_folder + '/label_stats.json', 'r') as file:
    stats = json.loads(file.read())

# with open('/rhome/msaee007/bigdata/pointnet_data/p1_real_data' + '/label_stats.json', 'r') as file:
#     stats = json.loads(file.read())


device = 'cuda'

selected_model = 'resnet'
results_file = base_folder+'/%s_weather_results_summary.csv' % selected_model
time_file = base_folder+'/%s_weather_time_summary.csv' % selected_model

# TODO: consider adding experiment ratio vs. quality and memory

base_parameters = {
    'set_abstractions': [
            {'ratio': 1.0, 'radius': 2.0, 'max_neighbors': 16, 'mlp': [2 + 1, 32, 32, 64]},
            {'ratio': 0.01, 'radius': 2.0, 'max_neighbors': 16, 'mlp': [2 + 64, 128, 128, 256]}
        ],
        'global_abstraction': {'mlp': [2 + 256, 512, 512, 1024]},
        'final_mlp': [1024, 512, 512, 128, 9],
        'dropout':0.1
}
base_parameters['set_abstractions'][0]['mlp'][0] = 2 + 3
base_parameters['final_mlp'][-1] = len(outputs)
model = PointNet(**base_parameters)
# model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/all_values_parametrized_synth_0.013106.ckpt', weights_only=True))
model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/all_values_parametrized_weather_0.207067.ckpt', weights_only=True))
model = model.to(device)
model.eval()

parameters_comb = list(product(['16','32','64'], ['0.025','0.05','0.1','0.25']))
d = 'uniform'
id = list(labels[d].keys())[0]
data_lists = [[], [], [], [], [], [], [], [], [], [], []]
row = labels[d][id]
data_path = row['dataset_id']
if '.csv' not in data_path:
    data_path +=  '.csv'
data = pd.read_csv(data_path)
points = data[['x', 'y']].values
print('Points shape: ', points.shape)
x = data['att'].values
x = x.reshape((x.shape[0], 1))
points = torch.from_numpy(points).to(torch.float32).to(device)
(hotspots_k, kvalue_r) = parameters_comb[0]
parameters = np.array([float(hotspots_k), float(kvalue_r)])
new_x = np.zeros((x.shape[0], x.shape[1] + len(parameters)))
new_x[:, 0] = x[:, 0] 
for p in range(len(parameters)):
    new_x[:, p+1] = parameters[p]
new_x = torch.from_numpy(new_x).to(torch.float32).to(device)
for i in range(len(data_lists)):
    duplicated_points = points.repeat(2**i, 1)
    noise = torch.randn_like(duplicated_points) * 0.001
    data = Data(pos=duplicated_points+noise,
        x=new_x.repeat(2**i, 1)
    )
    data_lists[i].append(data)
# create batches

progress_bar = tqdm(
    range(len(data_lists)),
)
batch_times = []
input_sizes = []
train_memory = []
infe_memory = []
# n_edges = []
complete = 0
for batch_idx in progress_bar:
    batch_data = Batch.from_data_list(data_lists[batch_idx]).to(device)
    if batch_idx == 0:
        # warmup step
        prediction = model(batch_data)
        with torch.no_grad():
            if complete == 0:
                try:
                    _train_memory, train_time = measure_memory_usage(model, batch_data, mode="train")
                except:
                    _train_memory, train_time = -1, -1
                    complete += 1
            else:
                _train_memory, train_time = (-1, -1)
            try:
                _infe_memory, infe_time = measure_memory_usage(model, batch_data, mode="inference")
            except:
                break
            batch_times.append(infe_time)
            train_memory.append(_train_memory)
            infe_memory.append(_infe_memory)
            input_sizes.append(batch_data.pos.shape[0])
            print(input_sizes[-1], train_memory[-1], infe_memory[-1], batch_times[-1])

pd.DataFrame({'input_size': input_sizes, 'prediction_time': batch_times, 'inference_max_memory': infe_memory, 'train_max_memory': train_memory}).to_csv('/rhome/msaee007/PointNet/01_data_synth/results_csv_summary/scalability.csv', index=False)