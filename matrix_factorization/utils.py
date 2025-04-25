from pandas import DataFrame
import pandas as pd
from tqdm import tqdm

def split_data(df: DataFrame, test_size=0.25, random_state=42):
    train_rows = []
    val_rows = []
    n_clients = len(df["client_id"].unique())

    # Group by client_id and split interactions per client
    for client_id, group in tqdm(df.groupby("client_id"), total=n_clients):
        if len(group) == 1:
            # Single interaction stays in train
            train_rows.append(group)
            continue

        # Determine how many go to val (at least 1, at most len-1)
        val_count = max(1, int(len(group) * test_size))
        val_count = min(val_count, len(group) - 1)

        # Sample validation indices
        val_idx = group.sample(n=val_count, random_state=random_state).index
        train_idx = group.index.difference(val_idx)

        train_rows.append(df.loc[train_idx])
        val_rows.append(df.loc[val_idx])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    val_df = pd.concat(val_rows).reset_index(drop=True)

    train_client_ids = train_df["client_id"].unique()
    val_client_ids = val_df["client_id"].unique()

    return train_df, val_df, train_client_ids, val_client_ids
