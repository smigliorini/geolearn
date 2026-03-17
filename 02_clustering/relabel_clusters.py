import math
import pandas as pd
import numpy as np
from glob import glob

input_path = '/rhome/msaee007/bigdata/weather_data/labeled/'

files = glob(input_path + '*.csv')

temp_mins = []
temp_maxs = []
temp_ranges = []
avg_cluster_temps = []
avg_cluster_temps2 = []
matching_cluster_labels = []

global_min_temp = 100000
global_max_temp = -100000

ttt = 0
for fpath in files:
    print(f"{ttt} of {len(files)}")
    ttt += 1
    df = pd.read_csv(fpath)
    df2 = df[df['TEMP'] < 900][df['LABELS'] != -1].reindex()
    global_min_temp = min(global_min_temp, df2['TEMP'].min())
    global_max_temp = max(global_max_temp, df2['TEMP'].max())
    df2 = df2[['TEMP','LABELS']].groupby('LABELS').mean()
    df2 = df2.sort_values('TEMP')
    if df2.shape[0] == 0:
        continue
    labels = {}
    for i in range(df2.shape[0]):
        old_label = df2.iloc[i].name
        labels[old_label] = i
    df.loc[:, 'NEW_LABEL'] = -1
    for l in labels:
        i = labels[l]
        df.loc[df['LABELS'] == l, 'NEW_LABEL'] = i
    df[['STATION', 'LONGITUDE', 'LATITUDE', 'ELEVATION', 'TEMP', 'NEW_LABEL']].to_csv(fpath.replace('labeled', 'new_labels'))
    # vals = df['TEMP'].values.tolist()
    # vals2 = [math.floor(v/10) for v in vals]
    # avg_cluster_temps += vals
    # avg_cluster_temps2 += vals2
    # matching_cluster_labels.append(len(np.unique(vals))  == len(np.unique(vals2)))
print("global_min_temp", global_min_temp)
print("global_min_temp", global_max_temp)
