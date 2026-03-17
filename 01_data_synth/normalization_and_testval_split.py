import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
import numpy as np

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
    

folder = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data/'
dists = ['uniform', 'diagonal', 'gaussian', 'sierpinski']

_files = {d: sorted([str(p) for p in Path(f'{folder}/{d}').glob(f'**/*.csv')]) for d in dists}

files = []
for d in _files:
    files += [p for p in _files[d] if 'data_summary' not in p]

labels_files = {d: str(Path(f'{folder}/{d}/data_summary.csv')) for d in dists}
labels = {}
for d in labels_files:
    labels[d] = read_labels(labels_files[d])

labels_lists = defaultdict(list)

for d in labels:
    for id in labels[d]:
        labels_lists['dist'].append(d)
        labels_lists['id'].append(id)
        t = 'hotspots'
        for k in labels[d][id][t]:
            for l in labels[d][id][t][k]:
                if 'time' == l:
                    continue
                labels_lists[f'hotspots_{k}_{l}'].append(labels[d][id][t][k][l])
        t = 'k_values'
        for k in labels[d][id][t]:
            labels_lists[f'k_value_{k}'].append(labels[d][id][t][k]['value'])
        labels_lists['e0'].append(labels[d][id]['box_counts']['e0'])
        labels_lists['e2'].append(labels[d][id]['box_counts']['e2'])


df = pd.DataFrame(labels_lists)
df.to_csv(f'{folder}/labels.csv', index=False)


train_samples = defaultdict(list)
val_samples = defaultdict(list)
for d in labels:
    ids = df[df['dist'] == d].id.values
    np.random.shuffle(ids)
    train = list(ids[:int(0.8*len(ids))])
    train_samples['id'] += train
    train_samples['dist'] += [d]*len(train)
    val = list(ids[int(0.8*len(ids)):])
    val_samples['id'] += val
    val_samples['dist'] += [d]*len(val)


train_samples = pd.DataFrame(train_samples)
train_samples.to_csv(f'{folder}/train_samples.csv', index=False)
val_samples = pd.DataFrame(val_samples)
val_samples.to_csv(f'{folder}/val_samples.csv', index=False)


# for c in #['k_value_0.025', 'k_value_0.05', 'k_value_0.1', 'k_value_0.25']:
    # stats[c] = {'mean': df[c].mean(), 'std': df[c].std(), 'min': df[c].min(), 'max': df[c].max()}

values = defaultdict(list)
for i,row in train_samples.iterrows():
    rr = df[(df['dist'] == row['dist']) & (df['id'] == row['id'])]
    for c in df.columns:
        if 'dist' in c or 'id' in c:
            continue
        v = rr[c].iloc[0]
        values[c].append(v)
        if 'k_value' in c:
            values['k_values'].append(v)
        elif 'min_val' in c:
            values['hotspots_min_val'].append(v)
        elif 'max_val' in c:
            values['hotspots_max_val'].append(v)
        elif 'min_x' in c:
            values['hotspots_min_x'].append(v)
        elif 'max_x' in c:
            values['hotspots_max_x'].append(v)
        elif 'min_y' in c:
            values['hotspots_min_y'].append(v)
        elif 'max_y' in c:
            values['hotspots_max_y'].append(v)
    # k_values += list(df[c].values)

stats = {}
for c in values:
    print(c)
    stats[c] = {'mean': np.mean(values[c]), 'std': np.std(values[c]), 'min': np.min(values[c]), 'max': np.max(values[c])}

with open(f'{folder}/label_stats.csv', 'w') as file:
    file.write(json.dumps(stats, indent=2))

