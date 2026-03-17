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

# ground truth time
def read_labels(path):
    with open(path, 'r') as file:
        lines = file.readlines()
        records = {}
        for l in lines:
            j = json.loads(l)
            id = j['dataset_id']
            id = id[id.rfind('/')+1:]
            if '.csv' in id:
                id = id[:id.rfind('.csv')]
            records[id] = j
        return records
    

base_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data'
val_samples = pd.read_csv(base_folder + '/val_samples.csv')

dists = ['uniform', 'diagonal', 'gaussian', 'sierpinski']
# labels_files = {d: str(Path(f'{base_folder}/{d}/data_summary.csv')) for d in dists}
# labels = {}
# for d in labels_files:
#     labels[d] = read_labels(labels_files[d])

# for d in dists:
#     for i in range(len(labels[d])):
#         dist, id = d, str(i)
#         label = labels[dist][id]
#         data = pd.read_csv(base_folder + f'/{dist}/{id}.csv')
#         points = data[['x', 'y', 'att']].values
#         tree = KDTree(points[:, [0,1]], leaf_size=1)
#         ks = [16, 32, 64]
#         for k in ks:
#             t1 = time.time()
#             _, knn_index = tree.query(points[:, [0,1]], k=k)
#             averages = np.zeros(knn_index.shape[0])
#             for p in range(len(averages)):
#                 averages[p] = points[knn_index[p], 2].sum()/len(knn_index[p])
#             arg_min = averages.argmin()
#             arg_max = averages.argmax()
#             t1 = time.time()-t1
#             label['hotspots'][str(k)]['time'] = t1
#         t1 = time.time()
#         get_histogram(64, points[:, [0,1]], points[:, 2].reshape((points.shape[0], 1)))
#         t1 = time.time()-t1
#         label['histogram_time'] = t1
#         with open(base_folder + f'/{dist}/data_summary2.csv', 'a') as file:
#             file.write(json.dumps(label) + '\n')


labels_files = {d: str(Path(f'{base_folder}/{d}/data_summary2.csv')) for d in dists}
labels = {}
for d in labels_files:
    labels[d] = read_labels(labels_files[d])

groun_truth_table = defaultdict(list)
histogram_total_time = 0.0
ground_truth_time = 0.0
for i in range(val_samples.shape[0]):
    row = val_samples.iloc[i]
    label = labels[row['dist']][str(row['id'])]
    groun_truth_table['dist'].append(row['dist'])
    groun_truth_table['id'].append(row['id'])
    groun_truth_table['tree_build_time'].append(label['tree_time'])
    groun_truth_table['hotspots_16_time'].append(label['hotspots']['16']['time'])
    groun_truth_table['hotspots_32_time'].append(label['hotspots']['32']['time'])
    groun_truth_table['hotspots_64_time'].append(label['hotspots']['64']['time'])
    groun_truth_table['k_values_0.025_time'].append(label['k_values']['0.025']['time'])
    groun_truth_table['k_values_0.05_time'].append(label['k_values']['0.05']['time'])
    groun_truth_table['k_values_0.1_time'].append(label['k_values']['0.1']['time'])
    groun_truth_table['box_counts_time'].append(label['box_counts']['time'])
    groun_truth_table['histogram_time'].append(label['histogram_time'])
    ground_truth_time += (label['tree_time'] +
                      label['hotspots']['16']['time'] + 
                      label['hotspots']['32']['time'] + 
                      label['hotspots']['64']['time'] + 
                      label['k_values']['0.025']['time'] + 
                      label['k_values']['0.05']['time'] + 
                      label['k_values']['0.1']['time'] + 
                      label['box_counts']['time']
            )
    histogram_total_time += label['histogram_time']
    
pd.DataFrame(groun_truth_table).to_csv(base_folder+'/ground_truth_time.csv',index=False)
print('Ground truth total time: ', ground_truth_time)
print('Histogram total time: ', histogram_total_time)



# pointnet model 
device='cuda'
outputs = ['hotspots_##_min_val', 'hotspots_##_min_x', 'hotspots_##_min_y',
           'hotspots_##_max_val', 'hotspots_##_max_x', 'hotspots_##_max_y',
           'k_value_$$',
            'e0', 'e2']
batch_size = 128
val_dataset = SpatialDataset(is_train=False, outputs=outputs, parametrized=True,histogram=False, with_noise=False,rotate=False)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
base_parameters = {
    'set_abstractions': [
            {'ratio': 0.75, 'radius': 0.1, 'max_neighbors': 16, 'mlp': [2 + 1, 32, 32, 64]},
            {'ratio': 0.5, 'radius': 0.1, 'max_neighbors': 16, 'mlp': [2 + 64, 128, 128, 256]}
    ],
    'global_abstraction': {'mlp': [2 + 256, 512, 512, 1024]},
    'final_mlp': [1024, 512, 512, 128, 9],
    'dropout':0.1
}
base_parameters['set_abstractions'][0]['mlp'][0] = 2 + val_dataset[0].x.shape[1]
base_parameters['final_mlp'][-1] = len(outputs)

model = PointNet(**base_parameters)
model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/all_values_parametrized_0.055334.ckpt', weights_only=True))
model = model.to(device)
model.eval()

num_batches = len(val_loader)


pointnet_time = 0.0
pointnet_batch_time = []
for batch_idx in range(num_batches):
    data = next(iter(val_loader)).to(device)
    with torch.no_grad():
        t1 = time.time()
        prediction = model(data)
        batch_time = time.time() - t1
        pointnet_time += batch_time
        pointnet_batch_time.append(batch_time)
print('PointNet prediction time: ', pointnet_time)
print(pointnet_batch_time)



# resnet model 
val_dataset = SpatialDataset(is_train=False, outputs=outputs, parametrized=True,histogram=True, with_noise=False,rotate=False)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

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
model.load_state_dict(torch.load(base_folder + '/main_exp_outputs/resnet_baseline_synth_0.077375.ckpt', weights_only=True))
model = model.to(device)
model.eval()

resnet_time = 0.0
resnet_batch_time = []
for batch_idx in range(num_batches):
    hists, params, y = next(iter(val_loader))
    hists, params, y = hists.to(device), params.to(device), y.to(device)
    hists = hists.reshape(hists.shape[0], 3, hists.shape[-1], hists.shape[-1])
    with torch.no_grad():
        t1 = time.time()
        prediction = model(hists, params)
        batch_time = time.time() - t1
        resnet_time += batch_time
        resnet_batch_time.append(batch_time)
print('ResNet prediction time: ', resnet_time)
print(resnet_batch_time)
