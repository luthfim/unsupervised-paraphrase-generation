import pandas as pd
import os
import csv
import numpy as np

path_data = '/home/luthfi/workspace/unsupervised-paraphrase-generation/data/id_data'
files = os.listdir(path_data)

dfs = [pd.read_csv(os.path.join(path_data, f)) for f in files if f.endswith('.csv')]
df = pd.concat(dfs)

path_out = os.path.join(path_data, 'raw.txt')
keep_cols = ['text']
df = df[keep_cols].drop_duplicates().reset_index(drop=True)

np.savetxt(path_out, df.values, fmt = "%s")

# df.to_csv(path_out, index=False, header=None, quoting=csv.QUOTE_NONE, quotechar="",  escapechar=" ")