from generator import generate_graph
import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
import random


def scale(c, minimum=0, maximum=1):
    return (c - c.min()) / (c.max() - c.min()) * (maximum-minimum) + minimum

def uniform(low, high, size):
    samples = np.random.uniform(low, high, size)
    np.random.shuffle(samples)
    return samples

def get_gaussian_feature(dist=None):
    choices = np.arange(0.05, 1, 0.1)
    a = np.random.choice(choices)
    b = a if dist == 'diagonal' else np.random.choice(choices)
    c = np.random.choice(choices)
    d = c if dist == 'diagonal' else np.random.choice(choices)
    while abs(a-c)+abs(b-d) < 0.9:
        c = np.random.choice(choices)
        d = c if dist == 'diagonal' else np.random.choice(choices)
    return "%.2f,%.2f,%.2f,%.2f,0.1" % (a,b,c,d)

distribution = sys.argv[1]
print(distribution)
REP = int(sys.argv[2])  # how many samples to generate per parameter set
df = pd.read_csv("./parameters3.csv")
df = df[df['distribution'] == distribution]
df = df.reset_index(drop=True)
df = df.drop('id', axis=1)
# df['id'] = np.arange(0, df.shape[0])
directory = '/rhome/msaee007/bigdata/pointnet_data/synthetic_data_test/' + distribution + '/'
if not os.path.exists(directory):
    os.makedirs(directory)


with open(directory + 'data_summary.csv', 'w') as file:
    file.write("")

# df['min'] = scale(np.random.normal(loc=0.5, scale=0.5, size=df.shape[0]), 0, 0.5)
# df['max'] = scale(np.random.normal(loc=0.5, scale=0.5, size=df.shape[0]), 0.5, 1.0)

# df['min'] = uniform(0.0, 0.5, df.shape[0])
# df['max'] = uniform(0.5, 1.0, df.shape[0])
# if (df['min'] == df['max']).any():
#     print('Some min equals max!!')
# df['max'][df['min'] == df['max']] += 0.00001

# print(df)
# df.to_csv(directory + 'datasets.csv', index=True, index_label='id')
# labels = np.zeros((df.shape[0] * REP, 13))
    
progress = tqdm(total=df.shape[0]*REP)
i = 0
for index, row in df.iterrows():
    for _ in range(REP):
        # if os.path.exists(directory + str(row['id'])+'_graph.png'):
        #     continue
        # labels[i, 0] = i
        # labels[i, 1:] = generate_graph(
        dis_min = random.uniform(0.0, 1.0)
        dis_max = random.uniform(0.0, 1.0)
        while abs(dis_min-dis_max) < 0.25:
            dis_min = random.uniform(0.0, 1.0)
            dis_max = random.uniform(0.0, 1.0)
        dis_min, dis_max = min(dis_min, dis_max), max(dis_min, dis_max)
        generate_graph(
            path=directory + "%d" % (i),  # the path where the output will be stored
            distribution=row['distribution'],
            cardinality=row['cardinality'],
            dimensions=2,
            geometryType=row['shape'] if 'shape' in row else 'point',
            percentage=row['percentage'] if 'percentage' in row else None,
            buffer=row['buffer'] if 'buffer' in row else None,
            split_range=row['split_range'] if 'split_range' in row else None,
            dither=row['dither'] if 'dither' in row else None,
            probability=row['probability'] if 'probability' in row else None,
            digits=row['digits'] if 'digits' in row else None,
            # affineMatrix=None if row['affineMatrix'] == "None" else row['affineMatrix'],
            # maxSize=None if row['maxSize'] == "None" else row['maxSize'],
            # minGaussianFeatureCenter=  "0.75,0.25,0.1", #row['minGaussianFeatureCenter'],
            # maxGaussianFeatureCenter= "0.25,0.75,0.1", #row['maxGaussianFeatureCenter'],
            gaussianFeature = get_gaussian_feature(distribution),
            seed=None,
            # graph_size=row['graphSize'],  # maximum number of nodes in the output graph
            # alpha=row['alpha'],  # the radius for the range query when building the graph
            # k=row['k'],  # number of neibhros for knn-join query, for dataset label values
            dis_min= dis_min, #row['min'],
            dis_max= dis_max  #row['max'],
            # plot_data=i % REP == 0 and row['cardinality'] == 10000
        )
        i += 1
        progress.update(1)
# labels_df = pd.DataFrame(labels, columns=[
#     "dataset_id", "k",  "avg_att_min_val", "avg_att_min_x", "avg_att_min_y", "avg_att_min_z",
#     "avg_att_max_val", "avg_att_max_x", "avg_att_max_y", "avg_att_max_z", "e0", "e2",
#     #"k_value_0.025", "k_value_0.050", "k_value_0.100", "k_value_0.250"
#     "k_value_0.050"
# ])
# labels_df.to_csv(directory + 'data_summary.csv', index=False)