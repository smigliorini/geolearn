import numpy as np
from sklearn.neighbors import KDTree
import time
import pandas as pd
import json
from glob import glob
from generator import box_counts2
from collections import defaultdict
from tqdm import tqdm
paths = glob('/rhome/msaee007/bigdata/weather_data/monthly_labeled/*.csv')
max_temp = 699
min_temp = -699

train_years = ['1899', '1931', '1925', '1972', '1889', '1830', '1897', '1845', '1864', '1882', '1827', '1917', '1954', '1896', '1905',
               '2006', '1937', '1898', '1904', '2017', '1869', '1915', '1901', '1836', '1968', '1942', '2011', '1828', '1853', '1926',
               '1847', '1990', '1872', '1902', '1829', '2001', '1865', '1935', '1833', '1849', '2022', '1913', '1918', '1888', '1975',
               '1850', '1826', '1984', '1837', '1957', '1921', '1855', '1857', '1911', '1934', '1945', '1831', '1874', '1950', '1839',
               '1973', '1959', '1940', '1988', '1930', '1856', '1923', '1996', '1750', '2020', '1870', '1994', '1983', '1863', '1851',
               '2004', '1991', '1966', '1859', '1879', '1938', '1933', '1932', '1955', '1875', '1894', '1843', '1924', '2003', '2007',
               '1976', '2023', '1993', '1989', '1943', '1906', '1992', '1964', '1910', '1884', '1868', '1970', '1941', '1834', '1986',
               '1939', '1846', '1946', '2013', '1844', '1951', '1912', '1987', '1963', '1893', '1944', '1969', '1891', '2012', '1861',
               '2021', '2024', '1956', '1997', '1873', '1903', '1978', '1980', '1995', '1998', '1936', '1877', '1871', '1947', '1962',
               '1920', '2009', '2010', '1967', '1895', '2014', '1854', '1840', '1832', '1981', '2008', '2019', '2005', '1848', '1825',
               '1949', '1880', '1907', '1908', '1883', '1867', '1961']

val_years   = ['2000', '1824', '1866', '1928', '1900', '1927', '1858', '1974', '1878', '1909', '1977', '1952', '1881', '1842', '1982',
               '1862', '1916', '1953', '1979', '2015', '1919', '1958', '1965', '1948', '1841', '1929', '2002', '1852', '1985', '2018',
               '1960', '1922', '2016', '1914', '1835', '1876', '1999', '1838', '1860', '1971']


labels_lists = defaultdict(list)
train_samples = defaultdict(list)
val_samples = defaultdict(list)

output_dir = '/rhome/msaee007/bigdata/pointnet_data/p1_real_data'

# summary_path = out_path[:out_path.rfind('/')+1] + 'data_summary.csv'
summary_path = f'{output_dir}/weather/data_summary.json'
with open(summary_path, 'w') as file:
    file.write("")

for fpath in tqdm(paths):
    year = fpath[fpath.rfind('/')+1:fpath.rfind('_')]
    month = fpath[fpath.rfind('_')+1:fpath.rfind('.')]
    out_path = f'{output_dir}/weather/{year}_{month}.csv'
    is_train = year in train_years
    is_val = year in val_years
    if not (is_train or is_val):
        continue
    labels_lists['id'].append(f'{year}_{month}')
    labels_lists['dist'].append('weather')
    
    if is_train:
        train_samples['id'].append(labels_lists['id'][-1])
        train_samples['dist'].append(labels_lists['dist'][-1])
    else:
        val_samples['id'].append(labels_lists['id'][-1])
        val_samples['dist'].append(labels_lists['dist'][-1])

    df = pd.read_csv(fpath)[['STATION', 'LONGITUDE', 'LATITUDE', 'TEMP']]
    df = df.rename(columns={'LONGITUDE': 'x', 'LATITUDE': 'y', 'TEMP': 'att'})
    df['att'] = (df['att']-min_temp)/(max_temp-min_temp)

    labels = {
        "dataset_id": out_path,
        "tree_time": -1,
        "hotspots": {

        },
        "k_values": {

        },
        "box_counts": {

        }
    }

    points = df[['x','y']].values
    e0, e2, t1 = box_counts2(points, f'weather_data_{year}_{month}', 0, 0.0, 0.0, '')
    labels['box_counts'] = {'e0': float(e0), 'e2': float(e2), 'time': float(t1)}
    labels_lists['e0'].append(e0)
    labels_lists['e2'].append(e2)
    t1=time.time()
    tree = KDTree(points, leaf_size=1)
    labels['tree_time'] = time.time()-t1

    radii = [0.025, 0.05, 0.1, 0.25]
    n = points.shape[0]
    for r in radii:
        t1 = time.time()
        indices = tree.query_radius(points, r=r, return_distance=False)
        value = sum([len(indices[i]) for i in range(n)]) / (n * (n - 1))
        t1 = time.time()-t1
        labels['k_values'][r] = {'value': float(value), 'time': float(t1)}
    for r in labels['k_values']:
        labels_lists[f'k_value_{r}'].append(labels['k_values'][r]['value'])
    ks = [16, 32, 64]
    for k in ks:
        t1 = time.time()
        _, knn_index = tree.query(points, k=k)
        averages = np.zeros(knn_index.shape[0])
        for p in range(len(averages)):
            averages[p] = df.loc[knn_index[p], 'att'].mean()
        arg_min = averages.argmin()
        arg_max = averages.argmax()
        t1 = time.time() - t1
        labels["hotspots"][k] = {
            "min_val": float(averages[arg_min]),
            "min_x": float(points[arg_min, 0]),
            "min_y":float(points[arg_min, 1]),
            "max_val": float(averages[arg_max]),
            "max_x": float(points[arg_max, 0]),
            "max_y": float(points[arg_max, 1]),
            'time': float(t1)
        }
        for l in labels['hotspots'][k]:
            labels_lists[f'hotspots_{k}_{l}'].append(labels['hotspots'][k][l])
    df.to_csv(out_path, index=False)
    with open(summary_path, 'a') as file:
        file.write(json.dumps(labels) + '\n')


# def get_value(s):
#     return json.loads(s)['value']

# for c in df.columns:
#     if 'k_value' not in c:
#         continue
#     print(c)
#     df[c] = df[c].apply(get_value)

df = pd.DataFrame(labels_lists)
df.to_csv(f'{output_dir}/labels.csv', index=False)

train_samples = pd.DataFrame(train_samples)
train_samples.to_csv(f'{output_dir}/train_samples.csv', index=False)

val_samples = pd.DataFrame(val_samples)
val_samples.to_csv(f'{output_dir}/val_samples.csv', index=False)


merged = pd.merge(train_samples,df, on=['dist','id'], how='left')
stats = merged.describe().to_dict()

values = defaultdict(list)
for i,row in train_samples.iterrows():
    rr = df[(df['dist'] == row['dist']) & (df['id'] == row['id'])]
    for c in df.columns:
        if 'dist' in c or 'id' in c:
            continue
        v = rr[c].iloc[0]
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

# stats = {}
for c in values:
    # print(c)
    stats[c] = {'mean': float(np.mean(values[c])), 'std': float(np.std(values[c])), 'min': float(np.min(values[c])), 'max': float(np.max(values[c]))}

with open(f'{output_dir}/label_stats.json', 'w') as file:
    file.write(json.dumps(stats, indent=2))

