import pandas as pd
import os

def load_measurements(tcp_variant, base_path='ml/data'):
    folder = os.path.join(base_path, tcp_variant)
    all_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

    dfs = []
    for file in all_files:
        df = pd.read_csv(os.path.join(folder, file), sep=';')  
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
