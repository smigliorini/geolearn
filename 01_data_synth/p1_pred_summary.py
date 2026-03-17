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
import sys

data_collection = sys.argv[1]
selected_model = sys.argv[2]

from tqdm import tqdm
# ground truth time
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

results_file = base_folder+'/%s_results_summary.csv' % selected_model
time_file = base_folder+'/%s_time_summary.csv' % selected_model
if 'weather' in data_collection:
    results_file = base_folder+'/%s_weather_results_summary.csv' % selected_model
    time_file = base_folder+'/%s_weather_time_summary.csv' % selected_model
    
model = None
if selected_model == 'pointnet':
    base_parameters = {
    'set_abstractions': [
            {'ratio': 0.75, 'radius': 0.1, 'max_neighbors': 16, 'mlp': [2 + 1, 32, 32, 64]},
            {'ratio': 0.5, 'radius': 0.1, 'max_neighbors': 16, 'mlp': [2 + 64, 128, 128, 256]}
        ],
        'global_abstraction': {'mlp': [2 + 256, 512, 512, 1024]},
        'final_mlp': [1024, 512, 512, 128, 9],
        'dropout':0.1
    }
    base_parameters['set_abstractions'][0]['mlp'][0] = 2 + 3
    base_parameters['final_mlp'][-1] = len(outputs)
    model = PointNet(**base_parameters)
    if 'weather' in data_collection:
        model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/all_values_parametrized_weather_0.212609.ckpt', weights_only=True))
    else:
        model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/all_values_parametrized_synth_0.014546.ckpt', weights_only=True))
    model = model.to(device)
    model.eval()
else:
    class ResNet(nn.Module):
        def __init__(self, output_dim, parameters_dim=2):
            super(ResNet, self).__init__()
            # Load the ResNet model from Hugging Face
            config = ResNetConfig(num_channels=3, layer_type='basic', depths=[2,2], hidden_sizes=[128,256], embedding_size=16)
            self.resnet = ResNetModel(config)
            # Dimension of ResNet output and the extra feature dimension
            resnet_output_dim = config.hidden_sizes[-1]  # This is typically 2048 for ResNet-50
            combined_dim = resnet_output_dim + parameters_dim
            # Define the fully connected layer to produce the desired output dimension
            self.fc = nn.Linear(combined_dim, output_dim)
        def forward(self, image, extra_feature):
            # Pass the image through ResNet to get the pooled output
            resnet_features = self.resnet(image).pooler_output  # shape: (batch_size, resnet_output_dim)
            resnet_features = resnet_features.reshape((resnet_features.shape[0], resnet_features.shape[1]))
            # Concatenate the ResNet output with the extra features
            combined_features = torch.cat((resnet_features, extra_feature), dim=1)  # shape: (batch_size, combined_dim)
            # Pass through the fully connected layer to get the final output
            output = self.fc(combined_features)
            return output
    model = ResNet(len(outputs), 2).to(device)
    if 'weather' in data_collection:
        model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/resnet_weather_0.259091.ckpt', weights_only=True))
    else:
        model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/resnet_synth_0.055634.ckpt', weights_only=True))
    model = model.to(device)
    model.eval()


def get_pointnet_outputs(points, x, rep=5):
    times = []
    data = Data(pos=points,
                    x=x,
                    batch=torch.zeros(points.shape[0]).to(torch.int64).to(device)
                )
    for i in range(rep):
        t1 = time.time()
        outs = model(data)
        t1 = time.time()-t1
        times.append(t1)
    times = sorted(times)
    return [times[rep//2], *[v.item() for v in outs[0]]]

def get_resnet_outputs(histogram, parameters, rep=5):
    times = []
    histogram = histogram.unsqueeze(0)
    parameters = parameters.unsqueeze(0)
    for i in range(rep):
        t1 = time.time()
        outs = model(histogram, parameters)
        t1 = time.time()-t1
        times.append(t1)
    times = sorted(times)
    return [times[rep//2], *[v.item() for v in outs[0]]]

# Data lists
results_table = defaultdict(list)
time_table = defaultdict(list)
parameters_comb = list(zip(['16','32','64', '16'], ['0.025','0.05','0.1','0.25']))

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
pred_time = 0.0
for d in labels:
    # if 'weather' not in d:
    #     continue
    _results_table = defaultdict(list)
    time_table['dist'].append(d)
    data_list = []
    for id in labels[d]:
        row = labels[d][id]
        data_path = row['dataset_id']
        if '.csv' not in data_path:
            data_path +=  '.csv'
        data = pd.read_csv(data_path)
        points = data[['x', 'y']].values
        x = data['att'].values
        x = x.reshape((x.shape[0], 1))
        histogram = torch.from_numpy(get_histogram(64, points, x)).to(torch.float32).to(device)
        points = torch.from_numpy(points).to(torch.float32).to(device)
        for (hotspots_k, kvalue_r) in parameters_comb:
            parameters = np.array([float(hotspots_k), float(kvalue_r)])
            new_x = np.zeros((x.shape[0], x.shape[1] + len(parameters)))
            new_x[:, 0] = x[:, 0] 
            for p in range(len(parameters)):
                new_x[:, p+1] = parameters[p]
            new_x = torch.from_numpy(new_x).to(torch.float32).to(device)
            data = None
            if selected_model == 'pointnet':
                #outs = get_pointnet_outputs(points, new_x)
                data = Data(pos=points,
                    x=new_x
                )
            else:
                parameters = torch.from_numpy(parameters).to(torch.float32).to(device)
                data = (histogram, parameters)
                # outs = get_resnet_outputs(histogram, parameters)
            _results_table['dist'].append(d)
            _results_table['id'].append(id)
            _results_table['n_points'].append(points.shape[0])
            _results_table['hotspots_k'].append(hotspots_k)
            _results_table['kvalue_r'].append(kvalue_r)
            # _results_table['predictions'].append([])
            _results_table['hotspots_min_val'].append(None)
            _results_table['hotspots_min_x'].append(None)
            _results_table['hotspots_min_y'].append(None)
            _results_table['hotspots_max_val'].append(None)
            _results_table['hotspots_max_x'].append(None)
            _results_table['hotspots_max_y'].append(None)
            _results_table['k_values'].append(None)
            _results_table['box_counts_e0'].append(None)
            _results_table['box_counts_e2'].append(None)
            _results_table['time'].append(None)
            data_list.append(data)
    # create batches
    dist_time = 0.0
    if selected_model == 'pointnet':
        max_nodes = 1807203#1000000*16
        node_sizes = {}
        for i in range(len(data_list)):
            node_sizes[i] = _results_table['n_points'][i]
        batch_indexes = binpacking.to_constant_volume(node_sizes,max_nodes)
        batches = []
        for batch_idx in range(len(batch_indexes)):
            sample_indexes = batch_indexes[batch_idx]
            batch_data = []
            _sample_indexes = []
            for i in sample_indexes:
                data = data_list[i]
                batch_data.append(data)
                _sample_indexes.append(i)
            # batch_data = [train_dataset[i] for i in sample_indexes]
            batch_data = Batch.from_data_list(batch_data)
            batches.append((batch_data, _sample_indexes))
    else:
        batch_size = 64
        batch_indexes = np.arange(len(data_list))
        np.random.shuffle(batch_indexes)
        r = len(batch_indexes) % batch_size
        remainder = batch_indexes[len(batch_indexes)-r:].tolist()
        batch_indexes = batch_indexes[:len(batch_indexes)-r].reshape(((len(batch_indexes)-r)//batch_size,batch_size)).tolist()
        batches = []
        if len(remainder):
            batch_indexes.append(remainder)
        for batch_idx in range(len(batch_indexes)):
            sample_indexes = batch_indexes[batch_idx]
            batch_data_hist = []
            batch_data_param = []
            for i in sample_indexes:
                histogram, param = data_list[i]
                batch_data_hist.append(histogram)
                batch_data_param.append(param)
            batches.append((default_collate(batch_data_hist), default_collate(batch_data_param), sample_indexes))
    progress_bar = tqdm(
        range(len(batches)),
        desc=f"Evaluating {d}" 
    )
    for batch_idx in progress_bar:
        if selected_model == 'pointnet':
            batch_data, sample_indexes = batches[batch_idx]
            batch_data = batch_data.to(device)
            with torch.no_grad():
                torch.cuda.synchronize()
                start_event.record()
                prediction = model(batch_data)
                end_event.record()
                torch.cuda.synchronize()
                pred_time = start_event.elapsed_time(end_event)/1000
            if batch_idx == 0: # run it again the first time only
                with torch.no_grad():
                    start_event.record()
                    prediction = model(batch_data)
                    end_event.record()
                    torch.cuda.synchronize()
                    pred_time = start_event.elapsed_time(end_event)/1000
            dist_time += pred_time
            predictions = prediction.cpu().tolist()
        else:
            histograms, params, sample_indexes = batches[batch_idx]
            histograms, params = histograms.to(device), params.to(device)
            with torch.no_grad():
                torch.cuda.synchronize()
                start_event.record()
                prediction = model(histograms, params)
                end_event.record()
                torch.cuda.synchronize()
                pred_time = start_event.elapsed_time(end_event)/1000
            if batch_idx == 0: # run it again the first time only
                with torch.no_grad():
                    start_event.record()
                    prediction = model(histograms, params)
                    end_event.record()
                    torch.cuda.synchronize()
                    pred_time = start_event.elapsed_time(end_event)/1000
            dist_time += pred_time
            predictions = prediction.cpu().tolist()

        for i in range(len(sample_indexes)):
            j = sample_indexes[i]
            # _results_table['predictions'][j] = predictions[i]
            _results_table['hotspots_min_val'][j] = predictions[i][0]
            _results_table['hotspots_min_x'][j] = predictions[i][1]
            _results_table['hotspots_min_y'][j] = predictions[i][2]
            _results_table['hotspots_max_val'][j] = predictions[i][3]
            _results_table['hotspots_max_x'][j] = predictions[i][4]
            _results_table['hotspots_max_y'][j] = predictions[i][5]
            _results_table['k_values'][j] = predictions[i][6]
            _results_table['box_counts_e0'][j] = predictions[i][7]
            _results_table['box_counts_e2'][j] = predictions[i][8]
            _results_table['time'][sample_indexes[i]] = pred_time/len(sample_indexes)
    for k in _results_table:
        results_table[k] += _results_table[k]
    time_table['time'].append(dist_time)      



pd.DataFrame(results_table).to_csv(results_file,index=False)
pd.DataFrame(time_table).to_csv(time_file)