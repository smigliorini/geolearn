import pandas as pd
import numpy as np

def wmape(ref, pred):
    e = 0.0
    if abs(ref).sum() == 0:
        e = 1e-3
    return (abs(ref - pred).sum()) / (abs(ref).sum()+e)



p = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/selectivity_exp_output/pointnet_selectivity_0.620420.csv').clip(lower=0).clip(upper=1.0)
r = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/selectivity_exp_output/resnet_selectivity_1.243503.csv').clip(lower=0).clip(upper=1.0)

p = p.sort_values(by='actual')
r = r.sort_values(by='actual_selectivity')

bins = [0.0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0]
p['bucket'] = pd.cut(p['actual'], bins=bins, include_lowest=True)
r['bucket'] = pd.cut(r['actual_selectivity'], bins=bins, include_lowest=True)

combined_wmape = {'bucket': [], 'resnet_wmape': [], 'pointnet_wmape': [], 'random_wmape': []}

bucket = []
for b in set(p['bucket'].values):
    combined_wmape['bucket'].append(b)
    p2 = p[p['bucket'] == b]
    combined_wmape['pointnet_wmape'].append(wmape(p2['actual'], p2['predicted']))
    r2 = r[r['bucket'] == b]
    combined_wmape['resnet_wmape'].append(wmape(r2['actual_selectivity'], r2['predicted_selectivity']))
    rand = np.random.uniform(0, 1, size=p2.shape[0])
    combined_wmape['random_wmape'].append(wmape(p2['actual'], rand))

df = pd.DataFrame(combined_wmape)
df_sorted = df.sort_values(by='bucket')
df.to_csv('/rhome/msaee007/PointNet/03_selectivity/results_summary_selectivity.csv', float_format="%.2e", index=False)

# p_wmape = p.groupby('bucket', observed=False).agg(
#     Avg_wmape=('wmape', 'mean'),
#     Record_Count=('wmape', 'count'),
# ).reset_index()


# r['bucket'] = pd.cut(r['actual_selectivity'], bins=[0.0, 1e-05, 0.0001, 0.001, 0.01, 0.1, 1], include_lowest=True)
# r['wmape'] = wmape(r['actual_selectivity'], r['predicted_selectivity'])
# r_wmape = r.groupby('bucket', observed=False).agg(
#     Avg_wmape=('wmape', 'mean'),
#     Record_Count=('wmape', 'count'),
# ).reset_index()

# df = pd.merge(p_wmape, r_wmape, on='bucket')[['bucket','Avg_wmape_x', 'Avg_wmape_y']].rename(columns={'Avg_wmape_x': 'PointNet', 'Avg_wmape_y': 'ResNet'})
# df.to_csv('/rhome/msaee007/PointNet/03_selectivity/results_summary.csv', float_format="%.2e",)
