# scripts/preprocessing.py
import pandas as pd

def compute_metrics(df: pd.DataFrame, sample_interval: float = 0.1) -> pd.DataFrame:
    """
    Enrich a DataFrame with the following columns:
    - timestamp (estimated)
    - throughput (bytes/second)
    - packets_sent, packets_acked, packets_lost
    - loss_rate (per interval)
    """
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Create estimated timestamp
    df = df.reset_index(drop=True)
    df['timestamp'] = df.index * sample_interval

    # Ensure correct data types
    df = df.astype({
        'bytes_sent': 'int',
        'bytes_acked': 'int',
        'bytes_retrans': 'int',
        'mss': 'int',
        'rtt': 'float'
    })

    # Compute deltas per interval
    delta_bytes_sent = df['bytes_sent'].diff()
    delta_bytes_acked = df['bytes_acked'].diff()
    delta_retrans = df['bytes_retrans'].diff()

    df['throughput'] = delta_bytes_acked / sample_interval
    df['packets_sent'] = delta_bytes_sent / df['mss']
    df['packets_acked'] = delta_bytes_acked / df['mss']
    df['packets_lost'] = df['packets_sent'] - df['packets_acked']
    df['loss_rate'] = delta_retrans / delta_bytes_sent.replace(0, pd.NA) 

    # Drop rows with NaNs (from diff)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df