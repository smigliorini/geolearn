from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from joblib import parallel_backend
from st_dbscan import ST_DBSCAN
import time
from itertools import product
import json
import os

files = glob('/rhome/msaee007/bigdata/weather_data/monthly_partitions/**/**/*.csv')

print(f'Files to process {len(files)}')

output_folder = '/rhome/msaee007/bigdata/weather_data/labels_parametrized/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

with open(f'/rhome/msaee007/bigdata/weather_data/labels_parametrized/fit_time.log', 'w') as file:
            file.write('eps1,eps2,min_count,year,month,time,n_clusters,counts\n')

def get_year_and_month(fpath):
    s = fpath[fpath.find('YEAR=')+5:]
    year = int(s[:4])
    month = int(fpath[fpath.find('MONTH=')+6:fpath.rfind('/')])
    return year, month


global_min_temp = 100000
global_max_temp = -100000
max_clusters = 0


options = list(product(
    [0.02, 0.03, 0.05],
    [50, 100, 200],
    [5, 20, 50]
))

for (eps1, eps2, min_samples) in options:
    folder_name = '%.2f_%d_%d' % (eps1, eps2, min_samples)
    output_folder = '/rhome/msaee007/bigdata/weather_data/labels_parametrized/%s' % folder_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
         
    with parallel_backend('threading', n_jobs=-1):
        for i in range(len(files)):
            df = pd.read_csv(files[i])
            if df['TEMP'].count() < 1024:
                continue
            df = df[~df['TEMP'].isna()]
            year, month = get_year_and_month(files[i])
            global_min_temp = min(global_min_temp, df['TEMP'].min())
            global_max_temp = max(global_max_temp, df['TEMP'].max())
            gdf = gpd.GeoDataFrame(
                df, geometry=gpd.points_from_xy(df.LONGITUDE, df.LATITUDE), crs="EPSG:4326"
            )
            gdf = gdf.to_crs(epsg=6933)
            gdf['LONGITUDE'] = gdf.geometry.x
            gdf['LATITUDE'] = gdf.geometry.y
            selected_cols = ['LONGITUDE', 'LATITUDE']
            mins = gdf[selected_cols].min()
            maxs = gdf[selected_cols].max()
            gdf[selected_cols] = (gdf[selected_cols]-mins)/(maxs-mins)
            data = gdf[['TEMP', 'LONGITUDE', 'LATITUDE']].values
            st_dbscan = ST_DBSCAN(eps1 = eps1, eps2 = eps2, min_samples = min_samples, metric='euclidean', n_jobs=-1)
            fit_time = time.time()
            st_dbscan.fit(data)
            fit_time = time.time()-fit_time

            gdf['_LABELS'] = st_dbscan.labels
            df2 = gdf[['TEMP','_LABELS']][gdf['_LABELS'] != -1].groupby('_LABELS').mean()
            df2 = df2.sort_values('TEMP')
            
            labels = {}
            for i in range(df2.shape[0]):
                old_label = df2.iloc[i].name
                labels[old_label] = i
            gdf.loc[:, 'LABEL'] = 0
            for l in labels:
                i = labels[l]
                gdf.loc[gdf['_LABELS'] == l, 'LABEL'] = i+1

            
            values, counts = np.unique(gdf['LABEL'], return_counts=True)
            max_clusters = max(max_clusters, len(values))
            with open(f'/rhome/msaee007/bigdata/weather_data/labels_parametrized/fit_time.log', 'a') as file:
                file.write(f'{eps1},{eps2},{min_samples},{year},{month},{fit_time},{len(values)},"{str(values)}#{str(counts)}"\n')

            pd.DataFrame({
                'STATION': gdf['STATION'],
                'TEMP': gdf['TEMP'],
                'LONGITUDE': gdf['LONGITUDE'],
                'LATITUDE': gdf['LATITUDE'],
                'LABEL': gdf['LABEL']
            }).to_csv(f'{output_folder}/{year}_{month}.csv')
        
with open(f'/rhome/msaee007/bigdata/weather_data/stats.json', 'w') as file:
    file.write('{"max_temp":%f, "min_temp":%f, "max_clusters": %d}' % (global_max_temp, global_min_temp, max_clusters))