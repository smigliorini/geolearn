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

output_folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/hyperparameter_outputs'
device = torch.device('cuda')
max_epochs = 10
batch_size=8

outputs = ['hotspots_64_min_val', 'hotspots_64_min_x',
                            'hotspots_64_min_y', 'hotspots_64_max_val', 'hotspots_64_max_x',
                            'hotspots_64_max_y', 'k_value_0.25',  'e0', 'e2']
train_loader = DataLoader(SpatialDataset(is_train=True, outputs=outputs), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(SpatialDataset(is_train=False, outputs=outputs), batch_size=batch_size, shuffle=True)



def train_step(model, optimizer, epoch):
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


def val_step(model, epoch, label, best_wmape):
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


base_parameters = {
    'set_abstractions': [
            {'ratio': 0.75, 'radius': 0.1, 'max_neighbors': 64, 'mlp': [2 + 1, 8, 8, 16]},
            {'ratio': 0.75, 'radius': 0.1, 'max_neighbors': 64, 'mlp': [2 + 16, 32, 32, 64]}
    ],
    'global_abstraction': {'mlp': [2 + 64, 128, 128, 256]},
    'final_mlp': [256, 128, 128, 32, 9],
    'dropout':0.1
}

def update_width(parameters, w):
    parameters['set_abstractions'][0]['mlp'] = [2+1] + [w*v for v in parameters['set_abstractions'][0]['mlp'][1:]]
    parameters['set_abstractions'][1]['mlp'] = [2+ parameters['set_abstractions'][0]['mlp'][-1]] + [w*v for v in parameters['set_abstractions'][1]['mlp'][1:]]
    parameters['global_abstraction']['mlp'] = [2 + parameters['set_abstractions'][1]['mlp'][-1]] + [w*v for v in parameters['global_abstraction']['mlp'][1:]]
    parameters['final_mlp'] = [w*v for v in parameters['final_mlp'][:-1]] + [9]
    return parameters

output = {}

widths = [1, 2, 4]
widths_wmape = []
for w in widths:
    parameters = update_width(copy.deepcopy(base_parameters), w)
    model = PointNet(**parameters).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )

    # for name, layer in model._modules.items():
    #     print("#####" + name + "####")
    #     for name, _layer in layer._modules.items():
    #         print(name, _layer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: ', pytorch_total_params)
    label = f'width_{w}'
    best_wmape = 100
    for i in range(1,max_epochs+1):
            train_step(model, optimizer, i)
            best_wmape = val_step(model, i, label, best_wmape)
    widths_wmape.append(best_wmape.item())
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache() 


best_width = 2**(np.argmin(widths_wmape))
output['wdiths'] = widths
output['widths_wmape'] = widths_wmape
output['best_width'] = best_width
print('widths: ', widths)
print('widths wmape: ', widths_wmape)
print('best_width: ', best_width)

ratios = [(.75, .75), (.75, 0.5), (.5, 0.5)]
output['ratios'] = ratios
ratios_wmape = [min(widths_wmape)]
for r1, r2 in ratios[1:]:
    parameters = update_width(copy.deepcopy(base_parameters), best_width)
    parameters['set_abstractions'][0]['ratio'] = r1
    parameters['set_abstractions'][1]['ratio'] = r2

    model = PointNet(**parameters).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )

    # for name, layer in model._modules.items():
    #     print("#####" + name + "####")
    #     for name, _layer in layer._modules.items():
    #         print(name, _layer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: ', pytorch_total_params)
    label = f'ratio_{r1}_{r2}'
    best_wmape = 100
    for i in range(1,max_epochs+1):
            train_step(model, optimizer, i)
            best_wmape = val_step(model, i, label, best_wmape)
    ratios_wmape.append(best_wmape.item())
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache() 

print('ratios: ', ratios)
print('ratios wmape: ', ratios_wmape)
best_ratio = ratios[np.argmin(ratios_wmape)]
print('best ratio: ', best_ratio)
output['ratios_wmape'] = ratios_wmape
output['best_ratio'] = best_ratio

radii = [0.1, 0.05, 0.025, 0.0125]
output['radii'] = radii
radii_wmape = [min(ratios_wmape)]

for r in radii[1:]:
    parameters = update_width(copy.deepcopy(base_parameters), best_width)
    parameters['set_abstractions'][0]['ratio'] = best_ratio[0]
    parameters['set_abstractions'][1]['ratio'] = best_ratio[1]
    parameters['set_abstractions'][0]['radius'] = r
    parameters['set_abstractions'][1]['radius'] = r

    model = PointNet(**parameters).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )

    # for name, layer in model._modules.items():
    #     print("#####" + name + "####")
    #     for name, _layer in layer._modules.items():
    #         print(name, _layer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: ', pytorch_total_params)
    label = f'radius_{r}'
    best_wmape = 100
    for i in range(1,max_epochs+1):
            train_step(model, optimizer, i)
            best_wmape = val_step(model, i, label, best_wmape)
    radii_wmape.append(best_wmape.item())
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache() 

print('radii: ', radii)
print('radii wmape: ', radii_wmape)
best_radius = radii[np.argmin(radii_wmape)]
print('best radius: ', best_radius)
output['radii_wmape'] = radii_wmape
output['best_radius'] = best_radius

max_neigh = [64, 32, 16]
output['max_neighbors'] = max_neigh

max_neigh_wmape = [min(radii_wmape)]


for k in max_neigh[1:]:
    parameters = update_width(copy.deepcopy(base_parameters), best_width)
    parameters['set_abstractions'][0]['ratio'] = best_ratio[0]
    parameters['set_abstractions'][1]['ratio'] = best_ratio[1]
    parameters['set_abstractions'][0]['radius'] = best_radius
    parameters['set_abstractions'][1]['radius'] = best_radius
    parameters['set_abstractions'][0]['max_neighbors'] = k
    parameters['set_abstractions'][1]['max_neighbors'] = k

    model = PointNet(**parameters).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0e-4
    )

    # for name, layer in model._modules.items():
    #     print("#####" + name + "####")
    #     for name, _layer in layer._modules.items():
    #         print(name, _layer)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: ', pytorch_total_params)
    label = f'max_neigh_{k}'
    best_wmape = 100
    for i in range(1,max_epochs+1):
            train_step(model, optimizer, i)
            best_wmape = val_step(model, i, label, best_wmape)
    max_neigh_wmape.append(best_wmape.item())
    del model
    del optimizer
    gc.collect()
    torch.cuda.empty_cache() 



print('max_neigh: ', max_neigh)
print('max_neigh wmape: ', max_neigh_wmape)
best_max_neigh = max_neigh[np.argmin(max_neigh_wmape)]
print('best max_neigh: ', best_max_neigh)
output['max_neigh_wmape'] = max_neigh_wmape
output['best_max_neighbors'] = best_max_neigh

print('\n\n\n\n\n')
print(output)


output = {'wdiths': [1, 2, 4], 'widths_wmape': [0.1803920716047287, 0.13181690871715546, 0.11730697005987167], 'best_width': 4, 'ratios': [(0.75, 0.75), (0.75, 0.5), (0.5, 0.5)], 'ratios_wmape': [0.11730697005987167, 0.11513862013816833, 0.10575534403324127], 'best_ratio': (0.5, 0.5), 'radii': [0.1, 0.05, 0.025, 0.0125], 'radii_wmape': [0.10575534403324127, 0.1302407830953598, 0.1237182766199112, 0.12727150321006775], 'best_radius': 0.1, 'max_neighbors': [64, 32, 16], 'max_neigh_wmape': [0.10575534403324127, 0.12317142635583878, 0.12070781737565994], 'best_max_neighbors': 64}



import matplotlib.pyplot as plt

# Enable LaTeX-style fonts or use a professional font like Times New Roman
plt.rcParams.update({
    "text.usetex": True,  # Enable LaTeX rendering for text
    "font.family": "serif",  # Use serif fonts
    "font.serif": ["Times New Roman"],  # Specify the serif font
    "axes.labelsize": 10,  # Font size for axis labels
    "axes.titlesize": 12,  # Font size for the title
    "xtick.labelsize": 8,  # Font size for x-tick labels
    "ytick.labelsize": 8,  # Font size for y-tick labels
    "legend.fontsize": 8,  # Font size for the legend
})

# Data for each parameter
parameters = [
    ('Width', [1, 2, 4], [0.1804, 0.1318, 0.1173]),
    ('Ratio', [(0.75, 0.75), (0.75, 0.5), (0.5, 0.5)], [0.1173, 0.1151, 0.1058]),
    ('Radius', [0.1, 0.05, 0.025, 0.0125], [0.1058, 0.1302, 0.1237, 0.1273]),
    ('Max Neighbors', [64, 32, 16], [0.1058, 0.1232, 0.1207])
]

# Flattened data for plotting
x_labels = []
wmape_values = []
best_so_far = []
current_best = float('inf')

for param_name, values, wmape_list in parameters:
    for value, wmape in zip(values, wmape_list):
        x_labels.append(f"{param_name}: {value}")
        wmape_values.append(wmape)
        current_best = min(current_best, wmape)
        best_so_far.append(current_best)

# Adjusted figure dimensions for single-column layout
plt.figure(figsize=(4.5, 3.5))  # Width and height in inches for single-column

# Plot
plt.plot(x_labels, wmape_values, marker='o', label=r'\textbf{WMAPE Value}', color='blue')
plt.plot(x_labels, best_so_far, linestyle='--', label=r'\textbf{Best WMAPE So Far}', color='green')

# Formatting
# plt.title(r'\textbf{Hyperparameter Tuning Progression}', fontsize=12)
plt.xlabel(r'\textbf{Parameter Values Tested}', fontsize=10)
plt.ylabel(r'\textbf{WMAPE}', fontsize=10)
plt.xticks(rotation=90, fontsize=8)
plt.grid(alpha=0.4)
plt.legend(fontsize=8)
plt.tight_layout()

# Show plot
plt.savefig('hyperparam.pdf')
# plt.show()