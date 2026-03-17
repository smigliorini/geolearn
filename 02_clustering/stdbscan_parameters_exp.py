from glob import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from joblib import parallel_backend
from st_dbscan import ST_DBSCAN
import time
from itertools import product
import json

files = glob('/rhome/msaee007/bigdata/weather_data/monthly_partitions/**/**/*.csv')

print(f'Files to process {len(files)}')
# colnames = ['STATION','LONGITUDE','LATITUDE','ELEVATION','TEMP']

with open(f'/rhome/msaee007/bigdata/weather_data/fit_time.log', 'w') as file:
            file.write('year,month,time,n_clusters,counts\n')

with open(f'/rhome/msaee007/bigdata/weather_data/dbscan_exp.log', 'w') as file:
            file.write('eps1,eps2,min_count,std,outliers_percent,n_clusters\n')


def get_year_and_month(fpath):
    s = fpath[fpath.find('YEAR=')+5:]
    year = int(s[:4])
    month = int(fpath[fpath.find('MONTH=')+6:fpath.rfind('/')])
    return year, month


global_min_temp = 100000
global_max_temp = -100000
max_clusters = 0


options = list(product(
    [0.02, 0.03, 0.04, 0.05, 0.01, 0.02, 0.05, 0.1, 0.15],
    [10, 20, 50, 100, 200]
    [5, 20, 50, 100]
))

results = {}
for opt in options:
    results[opt] = {
        'std': [],
        'outliers_percent': [],
        'n_clusters': [],
    }

print('Files count: ', len(files))
print('Options count: ', len(options))
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
        selected_cols = ['LONGITUDE', 'LATITUDE'] #, 'ELEVATION']
        mins = gdf[selected_cols].min()
        maxs = gdf[selected_cols].max()
        gdf[selected_cols] = (gdf[selected_cols]-mins)/(maxs-mins)
        data = gdf[['TEMP', 'LONGITUDE', 'LATITUDE']].values
        #gdf['ELEVATION'] = gdf['ELEVATION'].fillna(gdf['ELEVATION'].mean())
        for option in options:
            st_dbscan = ST_DBSCAN(eps1 = option[0], eps2 = option[1], min_samples = option[2], metric='euclidean', n_jobs=-1)
            # gdf2 = df[['AVGTMAX', 'LONGITUDE', 'LATITUDE', 'ELEVATION']]
            # gdf2['ELEVATION'] = gdf2['ELEVATION'].fillna(gdf2['ELEVATION'].mean())
            # print(gdf2.isna().sum())
            # data = gdf[['TEMP', 'LONGITUDE', 'LATITUDE', 'ELEVATION']].values
            
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
            gdf.loc[:, 'LABEL'] = -1
            for l in labels:
                i = labels[l]
                gdf.loc[gdf['_LABELS'] == l, 'LABEL'] = i

            
            values, counts = np.unique(gdf['LABEL'], return_counts=True)
            # minimize std-deviation of the counts
            std = float('nan')
            if len(counts) > 1:
                std = np.std(counts[1:])
            outliers_percent = (counts[0]/gdf.shape[0])
            results[opt]['std'].append(std)
            results[opt]['outliers_percent'].append(outliers_percent)
            results[opt]['n_clusters'].append(len(values)-1)
            # minimize the outliers
            # max_clusters = max(max_clusters, len(values))
            # with open(f'/rhome/msaee007/bigdata/weather_data/fit_time.log', 'a') as file:
            #     file.write(f'{year},{month},{fit_time},{len(values)},"{str(values)}#{str(counts)}"\n')

            # pd.DataFrame({
            #     'STATION': gdf['STATION'],
            #     'TEMP': gdf['TEMP'],
            #     'LONGITUDE': gdf['LONGITUDE'],
            #     'LATITUDE': gdf['LATITUDE'],
            #     # 'ELEVATION': gdf['ELEVATION'],
            #     'LABEL': gdf['LABEL']
            # }).to_csv(f'/rhome/msaee007/bigdata/weather_data/monthly_labeled/{year}_{month}.csv')
        
            with open(f'/rhome/msaee007/bigdata/weather_data/dbscan_exp.log', 'a') as file:
                file.write(f'{option[0]},{option[1]},{option[2]},{std},{outliers_percent},{len(values)-1}\n')

# with open(f'/rhome/msaee007/bigdata/weather_data/stats.json', 'w') as file:
#     file.write('{"max_temp":%f, "min_temp":%f, "max_clusters": %d}' % (global_max_temp, global_min_temp, max_clusters))

# with open(f'/rhome/msaee007/bigdata/weather_data/dbscan_exp_stats.json', 'w'):
#     file.write(json.dumps(results, indent=2))

df = pd.read_csv('/rhome/msaee007/bigdata/weather_data/dbscan_exp.log')
# mean_agg = df.groupby(['eps1', 'eps2', 'min_count']).mean()
max_agg = df.groupby(['eps1', 'eps2', 'min_count']).max()

# agg = agg[agg['n_clusters'] >= 4].copy()
# agg = agg[agg['std'] <= 1000].copy()
# agg = agg[agg['outliers_percent'] <= 0.15].copy()

# agg['score'] = (.5 * (agg['std']-agg['std'].min())/(agg['std'].max()-agg['std'].min())
#                 + .5 * (agg['outliers_percent']-agg['outliers_percent'].min())/(agg['outliers_percent'].max()-agg['outliers_percent'].min()))

max_agg['score'] = (1.0/3 * (max_agg['std']-max_agg['std'].min())/(max_agg['std'].max()-max_agg['std'].min())
               + 1.0/3 * (max_agg['outliers_percent']-max_agg['outliers_percent'].min())/(max_agg['outliers_percent'].max()-max_agg['outliers_percent'].min())
               + 1.0/3 * ((max_agg['n_clusters']-max_agg['n_clusters'].min())/(max_agg['n_clusters'].max()-max_agg['n_clusters'].min())))
name = max_agg.iloc[max_agg['score'].argmin()].name
print(df[(df['eps1'] == name[0]) & (df['eps2'] == name[1]) & (df['min_count'] == name[2])].describe())