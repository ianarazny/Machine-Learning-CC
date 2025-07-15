import pandas as pd
import glob
import os

def load_measurements(tcp_variant):
    path = f'../data/{tcp_variant}/'
    files = glob.glob(os.path.join(path, '*.csv'))
    dfs = [pd.read_csv(file) for file in files]
    return pd.concat(dfs, ignore_index=True)
