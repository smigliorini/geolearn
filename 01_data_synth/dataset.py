from torch_geometric.data import Data, Dataset
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import torch
from sklearn.neighbors import KDTree
import random

def get_histogram(histogram_size, points, x):
    histogram_min = np.full((histogram_size, histogram_size), np.inf)
    histogram_max = np.full((histogram_size, histogram_size), -np.inf)
    histogram_count = np.zeros((histogram_size, histogram_size))
    x_edges = np.linspace(0, 1, histogram_size + 1)
    y_edges = np.linspace(0, 1, histogram_size + 1)
    x_bin_indices = np.digitize(points[:, 0], x_edges) - 1
    y_bin_indices = np.digitize(points[:, 1], y_edges) - 1
    for i in range(points.shape[0]):
        x_bin = x_bin_indices[i]
        y_bin = y_bin_indices[i]
        if 0 <= x_bin < histogram_size and 0 <= y_bin < histogram_size:
            histogram_min[x_bin, y_bin] = min(x[i, 0], histogram_min[x_bin, y_bin])
            histogram_max[x_bin, y_bin] = max(x[i, 0], histogram_min[x_bin, y_bin])
            histogram_count[x_bin, y_bin] += 1
    histogram_min[np.isinf(histogram_min)] = 0.0
    histogram_max[np.isinf(histogram_max)] = 0.0
    histogram = np.zeros((3, histogram_size, histogram_size))
    histogram[0, :, :] = histogram_min
    histogram[1, :, :] = histogram_max
    histogram[2, :, :] = histogram_count / x.shape[0]
    return histogram

class SpatialDataset(Dataset):
    def __init__(self, folder='/rhome/msaee007/bigdata/pointnet_data/synthetic_data', is_train=True, outputs=None, parametrized=False, histogram=False, with_noise=False, rotate=False):
        super().__init__()
        self.cache = {}
        self.folder = folder
        all_outputs = ['hotspots_16_min_val', 'hotspots_16_min_x',
                            'hotspots_16_min_y', 'hotspots_16_max_val', 'hotspots_16_max_x',
                            'hotspots_16_max_y', 'hotspots_32_min_val', 'hotspots_32_min_x',
                            'hotspots_32_min_y', 'hotspots_32_max_val', 'hotspots_32_max_x',
                            'hotspots_32_max_y', 'hotspots_64_min_val', 'hotspots_64_min_x',
                            'hotspots_64_min_y', 'hotspots_64_max_val', 'hotspots_64_max_x',
                            'hotspots_64_max_y', 'k_value_0.025', 'k_value_0.05', 'k_value_0.1',
                            'k_value_0.25', 'e0', 'e2']
        if outputs == None:
            self.outputs = ['hotspots_64_min_val', 'hotspots_64_min_x',
                            'hotspots_64_min_y', 'hotspots_64_max_val', 'hotspots_64_max_x',
                            'hotspots_64_max_y', 'k_value_0.25',  'e0', 'e2']
        else:
            self.outputs = outputs
        
        self.parametrized = parametrized
        self.histogram = histogram
        self.with_noise = with_noise
        self.rotate=rotate
        self.rs = [0.025, 0.05, 0.1, 0.25]
        self.ks = [16, 32, 64]
        # dists = ['uniform', 'gaussian', 'diagonal', 'sierpinski']
        self.labels = pd.read_csv(folder + '/labels.csv')
        with open(folder + '/label_stats.json', 'r') as file:
            stats = json.loads(file.read())
        # for c in ['k_value_0.025', 'k_value_0.05', 'k_value_0.1', 'k_value_0.25', 'e0', 'e2']:
        #     self.labels[c] = (self.labels[c] - stats[c]['min'])/(stats[c]['max']-stats[c]['min'])
        for c in stats:
            # self.labels[c] = (self.labels[c] - stats[c]['min'])/(stats[c]['max']-stats[c]['min'])
            c2 = c
            if 'hotspots_m' in c and self.parametrized:
                c2 = 'hotspots' + c[c.find('_')+3:]
            if 'k_value' in c and self.parametrized:
                c2 = 'k_values'
            if c not in self.labels:
                continue
            self.labels[c] = (self.labels[c] - stats[c2]['mean'])/(stats[c2]['std'])
        samples_tag = '/train_samples.csv' if is_train else '/val_samples.csv'
        hotspots_k = '64'
        kvalue_r = '0.25'
        for o in self.outputs:
            if 'hotspots' in o:
                hotspots_k = o[o.find('_')+1:o.find('_')+3]
            elif 'k_value' in o:
                kvalue_r = o[o.rfind('_')+1:]
        if not parametrized:
            df = pd.read_csv(folder + samples_tag)
            df['##'] = hotspots_k
            df['$$'] = kvalue_r
            self.samples = df
        else:
            self.parametrized_hotspot = len([k for k in self.outputs if 'hotspots_##' in k]) > 0
            self.parameterized_kvalue = len([k for k in self.outputs if 'k_value_$$' in k]) > 0
            df = pd.read_csv(folder + samples_tag)
            df['join_key'] = 0
            if self.parametrized_hotspot:
                df = df.merge(pd.DataFrame({'##': ['16', '32', '64'], 'join_key': 0}), on='join_key', how='outer')
            else:
                df['##'] = hotspots_k
            if self.parameterized_kvalue:
                df = df.merge(pd.DataFrame({'$$': ['0.025', '0.05', '0.1', '0.25'], 'join_key': 0}), on='join_key', how='outer')
            else:
                df['$$'] = kvalue_r
            self.samples = df.drop('join_key', axis=1)

    def __len__(self):
        return self.len()
    def __getitem__(self, idx):
        return self.get(idx)

    def len(self):
        return len(self.samples)
    def get(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        row = self.samples.iloc[idx]
        dist = row.dist
        id = row.id
        hotspots_k = row['##']
        kvalue_r = row['$$']
        data = pd.read_csv(self.folder + f'/{dist}/{id}.csv')
        points = data[['x', 'y']].values
        x = data[['att']].values
        x = x.reshape((x.shape[0], 1))
        if self.rotate:
            rotation_angle = random.randrange(360)
            poi_rotated = gpd.points_from_xy(points[:, 0], points[:, 1]).rotate(rotation_angle, origin=(0.5, 0.5))
            points[:, 0] = np.array([p.x for p in poi_rotated])
            points[:, 1] = np.array([p.y for p in poi_rotated])
        parameters=np.array([])
        if self.parametrized:
            if self.parametrized_hotspot and self.parameterized_kvalue:
                outputs = [k.replace('##', hotspots_k).replace('$$', kvalue_r) for k in self.outputs]
                parameters = np.array([float(hotspots_k), float(kvalue_r)])
            elif self.parametrized_hotspot:
                outputs = [k.replace('##', hotspots_k) for k in self.outputs]
                parameters = np.array([float(hotspots_k)])
            else:
                outputs = [k.replace('$$', kvalue_r) for k in self.outputs]
                parameters = np.array([float(kvalue_r)])
        else:
            outputs = self.outputs
        y = self.labels[(self.labels['dist'] == dist) & (self.labels['id'] == id)][outputs]
        if self.with_noise:
            min_center = [y[f'hotspots_{int(hotspots_k)}_min_x'].values[0], y[f'hotspots_{int(hotspots_k)}_min_y'].values[0]]
            max_center = [y[f'hotspots_{int(hotspots_k)}_max_x'].values[0], y[f'hotspots_{int(hotspots_k)}_max_y'].values[0]]
            min_val = y[f'hotspots_{int(hotspots_k)}_min_val']
            max_val = y[f'hotspots_{int(hotspots_k)}_max_val']
            tree = KDTree(points, leaf_size=1)
            _, knn_index = tree.query([min_center, max_center], k=int(hotspots_k))
            center_points = knn_index.flatten()
            selected = [i for i in range(y.shape[0]) if i not in center_points]
            random.shuffle(selected)
            selected = selected[:round(len(selected)*0.1)]
            for i in selected:
                p = points[i]
                d1 = np.linalg.norm(p - min_center)
                d2 = np.linalg.norm(p - max_center)
                if d1 < d2:
                    x[i, 0] = max_val
                else:
                    x[i, 0] = min_val
        y = y.values
        if len(parameters) > 0:
            new_x = np.zeros((x.shape[0], x.shape[1] + len(parameters)))
            new_x[:, 0] = x[:, 0] 
            for i in range(len(parameters)):
                new_x[:, i+1] = parameters[i]
            x = new_x
        if self.histogram:
            histogram = get_histogram(64, points, x)
            out = (torch.from_numpy(histogram).to(torch.float32),
                    torch.from_numpy(parameters).to(torch.float32),
                    torch.from_numpy(y).to(torch.float32))
            self.cache[idx] = out
            return out
        else:
            data = Data(pos=torch.from_numpy(points).to(torch.float32),
                    x=torch.from_numpy(x).to(torch.float32),
                    y=torch.from_numpy(y).to(torch.float32))
            self.cache[idx] = data
            return data
