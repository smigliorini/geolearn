import pandas as pd
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

# Homogeneity: A cluster contains only points from a single class.
# h = 1 - \frac{H(C|K)}{H(C)}
# where  H(C|K)  is the conditional entropy of classes given clusters.
# Completeness: All points of a given class are assigned to the same cluster.
# c = 1 - \frac{H(K|C)}{H(K)}
# V-Measure: Harmonic mean of Homogeneity and Completeness.

from collections import defaultdict

pn1 = pd.read_csv('/rhome/msaee007/PointNet/02_density/clustering_results/pn_not_param.csv')
pn2 = pd.read_csv('/rhome/msaee007/PointNet/02_density/clustering_results/pn_param.csv')
un1 = pd.read_csv('/rhome/msaee007/PointNet/02_density/clustering_results/unet_not_param.csv')
un2 = pd.read_csv('/rhome/msaee007/PointNet/02_density/clustering_results/unet_param.csv')

data = {
    'pointnet_not_param': pn1,
    'unet_not_param': un1,
    'pointnet_param': pn2,
    'unet_param': un2
}

results = defaultdict(list)
for k in data:
    group_cols = [c for c in data[k].columns if 'param' in c or 'batch' in c]
    dfs = {group: data for group, data in data[k].groupby(group_cols)}
    for group in dfs:
        df = dfs[group]
        homogeneity = homogeneity_score(df['actual'], df['predicted'])
        completeness = completeness_score(df['actual'], df['predicted'])
        v_measure = v_measure_score(df['actual'], df['predicted'])
        results['model'].append(k)
        results['group'].append(group)
        params = group[1:]
        if len(params) == 0:
            params = (0.05, 200, 50)
        results['params'].append(params)
        results['homogeneity'].append(homogeneity)
        results['completeness'].append(completeness)
        results['v_measure'].append(v_measure)

results_df = pd.DataFrame(results)

summary = results_df.groupby(['model','params'])[['homogeneity',  'completeness',  'v_measure']].mean()

summary2 = results_df[~results_df['model'].str.contains('not_param')].groupby('model')[['homogeneity',  'completeness',  'v_measure']].mean()
summary2['params']  = 'mean'

summary = pd.concat([summary.reset_index(), summary2.reset_index()])
summary = summary.sort_values(by=['model', 'params'])

summary.to_csv('/rhome/msaee007/PointNet/02_density/clustering_results/clustering_agg_results.csv', index=False)


# time agg
dbscan_df = pd.read_csv('/rhome/msaee007/bigdata/weather_data/labels_parametrized/fit_time.log')

val_years   = ['2000', '1824', '1866', '1928', '1900', '1927', '1858', '1974', '1878', '1909', '1977', '1952', '1881', '1842', '1982',
               '1862', '1916', '1953', '1979', '2015', '1919', '1958', '1965', '1948', '1841', '1929', '2002', '1852', '1985', '2018',
               '1960', '1922', '2016', '1914', '1835', '1876', '1999', '1838', '1860', '1971']


params = [(0.05,200,50), (0.02,200,5),(0.03,50,5),(0.05,100,50),(0.03,200,20)]

dbscan_df = dbscan_df[dbscan_df['year'].isin([int(v) for v in val_years])]

dbscan_df = dbscan_df[dbscan_df[['eps1', 'eps2', 'min_count']].apply(tuple,axis=1).isin(params)]

dbscan_df.groupby(['eps1','eps2','min_count'])['n_clusters'].mean()
# dbscan_time = dbscan_df.groupby(['eps1','eps2','min_count'])['time'].sum().reset_index()

# selected_rows = dbscan_time[dbscan_time[['eps1', 'eps2', 'min_count']].apply(tuple,axis=1).isin(params)]

print('Total time for all five parameter sets:', selected_rows['time'].sum())

selected_rows = dbscan_time[dbscan_time[['eps1', 'eps2', 'min_count']].apply(tuple,axis=1).isin([(0.05,200,50)])]
print('Time for (0.05,200,50):', selected_rows['time'].sum())

