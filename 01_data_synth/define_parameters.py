import numpy as np
import pandas as pd

params = [
	# ["uniform", "gaussian", "diagonal", "sierpinski"], # distributions
    ["bit", "parcel"],
	["point"], # shapes
	[10000,15000,20000], # cardinalities # [25000, 50000, 75000, 100000]
	# ["", "100,0,0,0,100,0"], # affineMatricies for uniform and gaussian distributions
	# [0, 0.5,  1.0], # buffer for diagonal distribution
	# [0, 0.5, 1.0], # percentage for diagonal distribution
	[0.2, 0.8], # probability for bit distribution
	[32], # digits for bit distribution
    [0.1, 0.9], # split_range
    [0.1, 0.9], # dither
	# ["0.01,0.01"], # defines width and height for generated boxes
	# [1000], # desired graph size
	# [0.10], # alpha: percentage of dataset diagonal to consider for kNN
	# ["0.75,0.75,0.25,0.25,0.1", "0.75,0.25,0.25,0.75,0.1", "0.1,0.1,0.9,0.9,0.1", "0.1,0.1,0.5,0.9,0.1",
	#  "0.5,0.1,0.1,0.9,0.1", "0.1,0.9, 0.5,0.1,0.1", "0.9,0.9, 0.1,0.1,0.1", "0.85,0.2, 0.1,0.75,0.1"], # center for min, center for max, and variance
	# [5, 10, 20, 50], # kNN (neighbors for KNN-Join query)
]

# 0.75,0.75
# 0.25,0.25
# 0.75,0.25
# 0.25,0.75
# 0.1,0.1
# 0.9,0.9
# 0.5,0.9
# 0.5,0.1
# 0.1,0.9
# 0.85,0.2
# 0.1,0.75

csvHeader=[
			"distribution",
		   "shape",
		   "cardinality",
		   # "affineMatrix",
		#    "buffer",
		#    "percentage",
		   "probability",
		   "digits",
           "split_range",
           "dither",
		   # "maxSize",
		#    "graphSize",
		#    "alpha",
		   # "gaussianFeature",
		#    "k"
           ]

meshgrid_result = np.array(np.meshgrid(*params)).T.reshape(-1, len(params))
df = pd.DataFrame(meshgrid_result, columns=csvHeader)

# df.loc[df['distribution'] != 'diagonal', 'buffer'] = 0
# df.loc[df['distribution'] != 'diagonal', 'percentage'] = 0
df.loc[df['distribution'] != 'bit', 'probability'] = 0
df.loc[df['distribution'] != 'bit', 'digits'] = 0
df.loc[df['distribution'] != 'parcel', 'split_range'] = 0
df.loc[df['distribution'] != 'parcel', 'dither'] = 0
# df = df.loc[:, ~df.columns.duplicated()]
df  = df.drop_duplicates()
df.to_csv('parameters3.csv', index=True, index_label="id")
