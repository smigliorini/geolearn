import pandas as pd
from pathlib import Path
import json
import time
import numpy as np
from collections import defaultdict
from dataset import get_histogram

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


results_table = defaultdict(list)
for d in labels:
    for i in labels[d]:
        row = labels[d][i]
        data_path = row['dataset_id']
        if '.csv' not in data_path:
            data_path +=  '.csv'
        results_table['dist'].append(d)
        results_table['id'].append(i)
        results_table['tree_build_time'].append(row['tree_time'])
        results_table['hotspots_16_time'].append(row['hotspots']['16']['time'])
        results_table['hotspots_32_time'].append(row['hotspots']['32']['time'])
        results_table['hotspots_64_time'].append(row['hotspots']['64']['time'])
        results_table['k_values_0.025_time'].append(row['k_values']['0.025']['time'])
        results_table['k_values_0.05_time'].append(row['k_values']['0.05']['time'])
        results_table['k_values_0.1_time'].append(row['k_values']['0.1']['time'])
        results_table['k_values_0.25_time'].append(row['k_values']['0.25']['time'])
        results_table['box_counts_time'].append(row['box_counts']['time'])
        data = pd.read_csv(data_path)
        points = data[['x', 'y', 'att']].values
        t1 = time.time()
        hist = get_histogram(64, points[:, [0,1]], points[:, 2].reshape((points.shape[0], 1)))
        t1 = time.time()-t1
        results_table['histogram_time'].append(t1)
        results_table['hotspots_16_min_val'].append(row['hotspots']['16']['min_val'])
        results_table['hotspots_16_min_x'].append(row['hotspots']['16']['min_x'])
        results_table['hotspots_16_min_y'].append(row['hotspots']['16']['min_y'])
        results_table['hotspots_16_max_val'].append(row['hotspots']['16']['max_val'])
        results_table['hotspots_16_max_x'].append(row['hotspots']['16']['max_x'])
        results_table['hotspots_16_max_y'].append(row['hotspots']['16']['max_y'])
        results_table['hotspots_32_min_val'].append(row['hotspots']['32']['min_val'])
        results_table['hotspots_32_min_x'].append(row['hotspots']['32']['min_x'])
        results_table['hotspots_32_min_y'].append(row['hotspots']['32']['min_y'])
        results_table['hotspots_32_max_val'].append(row['hotspots']['32']['max_val'])
        results_table['hotspots_32_max_x'].append(row['hotspots']['32']['max_x'])
        results_table['hotspots_32_max_y'].append(row['hotspots']['32']['max_y'])
        results_table['hotspots_64_min_val'].append(row['hotspots']['64']['min_val'])
        results_table['hotspots_64_min_x'].append(row['hotspots']['64']['min_x'])
        results_table['hotspots_64_min_y'].append(row['hotspots']['64']['min_y'])
        results_table['hotspots_64_max_val'].append(row['hotspots']['64']['max_val'])
        results_table['hotspots_64_max_x'].append(row['hotspots']['64']['max_x'])
        results_table['hotspots_64_max_y'].append(row['hotspots']['64']['max_y'])
        results_table['k_values_0.025'].append(row['k_values']['0.025']['value'])
        results_table['k_values_0.05'].append(row['k_values']['0.05']['value'])
        results_table['k_values_0.1'].append(row['k_values']['0.1']['value'])
        results_table['k_values_0.25'].append(row['k_values']['0.25']['value'])
        results_table['box_counts_e0'].append(row['box_counts']['e0'])
        results_table['box_counts_e2'].append(row['box_counts']['e2'])
        results_table['n_points'].append(points.shape[0])


pd.DataFrame(results_table).to_csv(base_folder+'/ground_truth.csv',index=False)