import pandas as pd
from pathlib import Path
import json
import time
from sklearn.neighbors import KDTree
import numpy as np
from collections import defaultdict
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
from torch_geometric.data import Data
from itertools import product
import binpacking
from torch_geometric.data import Batch
from torch.utils.data.dataloader import default_collate
import sys
import os

sys.path.append(os.path.abspath("../01_data_synth/"))

from dataset import SpatialDataset, get_histogram
from pointnet import PointNet
from GeometryEncoder import GeometryEncoder
from p1_poly2vec_transformer import SimpleTransformerModel, pad_or_random_sample
data_collection = sys.argv[1]
# selected_model = sys.argv[2]

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

labels_files['bit'] = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/bit/data_summary.csv'
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

selected_model = 'poly2vec'
results_file = base_folder+'/%s_results_summary.csv' % selected_model
time_file = base_folder+'/%s_time_summary.csv' % selected_model
if 'weather' in data_collection:
    results_file = base_folder+'/%s_weather_results_summary.csv' % selected_model
    time_file = base_folder+'/%s_weather_time_summary.csv' % selected_model
    
geometry_model = GeometryEncoder(device).to(device)
model = SimpleTransformerModel(geometry_model)
if 'weather' in data_collection:
    model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/poly2vec_weather_0.463218.ckpt', weights_only=True, map_location=device))
else:
    model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/poly2vec_synth_0.145018.ckpt', weights_only=True, map_location=device))

model = model.to(device)
model.eval()



# Data lists
results_table = defaultdict(list)
time_table = defaultdict(list)
parameters_comb = list(zip(['16','32','64', '16'], ['0.025','0.05','0.1','0.25']))

# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)
pred_time = 0.0
for d in labels:
    # if 'weather' not in d:
    #     continue
    if 'weather' in data_collection and 'weather' not in d:
        continue
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
        pos = torch.from_numpy(points).to(torch.float32).to(device)
        for (hotspots_k, kvalue_r) in parameters_comb:
            parameters = np.array([float(hotspots_k), float(kvalue_r)])
            new_x = np.zeros((x.shape[0], x.shape[1] + len(parameters)))
            new_x[:, 0] = x[:, 0] 
            for p in range(len(parameters)):
                new_x[:, p+1] = parameters[p]
            new_x = torch.from_numpy(new_x).to(torch.float32).to(device)            
            points, new_x = pad_or_random_sample(pos, new_x)
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
            data_list.append((points,new_x))
    # create batches
    dist_time = 0.0
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
        batch_data_pos = []
        batch_data_x = []
        for i in sample_indexes:
            pos, x = data_list[i]
            batch_data_pos.append(pos)
            batch_data_x.append(x)
        batches.append((default_collate(batch_data_pos), default_collate(batch_data_x), sample_indexes))
    progress_bar = tqdm(
        range(len(batches)),
        desc=f"Evaluating {d}" 
    )
    for batch_idx in progress_bar:
        pos, x, sample_indexes = batches[batch_idx]
        pos, x = pos.to(device), x.to(device)
        with torch.no_grad():
            # torch.cuda.synchronize()
            # start_event.record()
            prediction = model(pos, torch.tensor([]).to(device), "points", x)
            # end_event.record()
            # torch.cuda.synchronize()
            # pred_time = start_event.elapsed_time(end_event)/1000
        if batch_idx == 0: # run it again the first time only
            # with torch.no_grad():
            #     start_event.record()
                prediction = model(pos, torch.tensor([]).to(device), "points", x)
                # end_event.record()
                # torch.cuda.synchronize()
                # pred_time = start_event.elapsed_time(end_event)/1000
        # dist_time += pred_time
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
# pd.DataFrame(time_table).to_csv(time_file)