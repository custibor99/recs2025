import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from baseline.aggregated_features_baseline.calculators import Calculator


def create_user_item_matrix(
    interactions_df: pd.DataFrame, relevant_client_ids: np.ndarray
) -> csr_matrix:
    user_id_to_index = {client_id: i for i, client_id in enumerate(relevant_client_ids)}
    n_users = len(relevant_client_ids)

    # Filter interactions to only include relevant clients
    interactions_df = interactions_df[
        interactions_df["client_id"].isin(user_id_to_index)
    ]

    # Identify all unique items (SKUs in this case)
    unique_items = interactions_df["sku"].unique()
    item_id_to_index = {sku: i for i, sku in enumerate(unique_items)}
    n_items = len(unique_items)

    # Calculate interaction counts per user-item pair
    interaction_counts = (
        interactions_df.groupby(["client_id", "sku"]).size().reset_index(name="count")
    )

    # Map client_ids and skus to their matrix indices
    row_indices = interaction_counts["client_id"].map(user_id_to_index).values
    col_indices = interaction_counts["sku"].map(item_id_to_index).values
    data_values = interaction_counts["count"].values

    # Create the sparse matrix (CSR format is good for computations)
    user_item_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)), shape=(n_users, n_items)
    )

    print(f"User-Item Matrix shape: {user_item_matrix.shape}")
    print(f"Sparsity: {user_item_matrix.nnz / (n_users * n_items):.6f}")

    return user_item_matrix


class SVDCalculator(Calculator):
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    @property
    def features_size(self) -> int:
        # not needed
        return -1

    def compute_features(self, user_item_matrix) -> np.ndarray:
        svd_model = TruncatedSVD(
            n_components=self.embedding_dim, n_iter=10, random_state=42
        )

        user_embeddings_float64 = svd_model.fit_transform(user_item_matrix)

        # Convert embeddings to float16 as required by the competition
        user_embeddings_float16 = user_embeddings_float64.astype(np.float16)

        print(f"Final embedding dtype: {user_embeddings_float16.dtype}")
        print(f"Final embedding shape: {user_embeddings_float16.shape}")

        return user_embeddings_float16
