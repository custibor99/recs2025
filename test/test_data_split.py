import pytest
import pandas as pd
from matrix_factorization.utils import split_data


def create_test_df():
    return pd.DataFrame(
        {
            "client_id": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 6, 6],
            "sku": [0, 101, 102, 101, 103, 102, 104, 105, 106, 107, 6, 6, 6],
        }
    )


def test_no_new_clients_in_val():
    df = create_test_df()
    train_df, val_df, train_clients, val_clients = split_data(df)
    # Every val client must also be a train client
    assert set(val_clients).issubset(set(train_clients)), (
        f"Validation clients {val_clients} not subset of train clients {train_clients}"
    )


def test_total_rows_preserved():
    df = create_test_df()
    train_df, val_df, _, _ = split_data(df)
    assert len(train_df) + len(val_df) == len(df), "Total row count not preserved"


def test_splits_non_empty():
    df = create_test_df()
    train_df, val_df, _, _ = split_data(df)
    assert len(train_df) > 0, "Training split is empty"
    assert len(val_df) > 0, "Validation split is empty"


def test_each_client_in_train():
    df = create_test_df()
    train_df, val_df, train_clients, _ = split_data(df)
    all_clients = df["client_id"].unique()
    # Every client must appear in train
    assert set(all_clients) == set(train_clients), (
        "Not every client_id is represented in the training split"
    )
