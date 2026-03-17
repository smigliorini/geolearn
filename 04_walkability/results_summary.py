import pandas as pd
import numpy as np

p = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/outputs/walkability_pointnet_0.148570.csv')
r = pd.read_csv('/rhome/msaee007/bigdata/pointnet_data/outputs/walkability_resnet_0.082551.csv')
p['predicted'] = p['predicted'].clip(lower=0).clip(upper=1.0)
r['predicted'] = r['predicted'].clip(lower=0).clip(upper=1.0)
p['Label'] = p['partition_label'].map(lambda x: x[:x.find('_')])
r['Label'] = r['label'].map(lambda x: x[:x.find('_')])

# p['InRange'] = (p['actual'] - p['predicted']) ** 2
# r['InRange'] = (r['actual'] - r['predicted']) ** 2
p['InRange'] = abs(p['actual'] - p['predicted']) < 0.1
r['InRange'] = abs(r['actual'] - r['predicted']) < 0.1

bins = np.arange(0,1.1,0.1)

p['bucket'] = pd.cut(p['actual'], bins=bins, include_lowest=True)
r['bucket'] = pd.cut(r['actual'], bins=bins, include_lowest=True)

p_InRange = p.groupby(['Label','bucket'], observed=False)['InRange'].mean().reset_index()
p_total_mean = p.groupby('Label')['InRange'].mean().reset_index()
p_total_mean['bucket'] = 'Total'
p_InRange = pd.concat([p_InRange, p_total_mean])
r_InRange = r.groupby(['Label','bucket'], observed=False)['InRange'].mean().reset_index()
r_total_mean = r.groupby('Label')['InRange'].mean().reset_index()
r_total_mean['bucket'] = 'Total'
r_InRange = pd.concat([r_InRange, r_total_mean])

df = pd.merge(p_InRange, r_InRange, on=['Label', 'bucket']).rename(columns={'InRange_x': 'PointNet', 'InRange_y': 'ResNet'})


melted_df = df.melt(id_vars=['Label', 'bucket'], var_name='Model', value_name='InRange')

# Pivot the DataFrame for LaTeX formatting
pivot_df = melted_df.pivot(index=['Label', 'Model'], columns='bucket', values='InRange').reset_index()


pivot_df.round(2).to_csv('/rhome/msaee007/PointNet/04_walkability/results_summary_walkability.csv', index=False)
# Convert to LaTeX format
# latex_table = pivot_df.to_latex(index=False, column_format="l l " + "r" * (len(pivot_df.columns) - 2))

# Print LaTeX table
# print(latex_table)