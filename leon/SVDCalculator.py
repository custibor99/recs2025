import numpy as np
import logging
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

from baseline.aggregated_features_baseline.calculators import Calculator

INTERACTION_WEIGHTS = {
    "product_buy": 5.0,
    "add_to_cart": 1.0,
    "remove_from_cart": -0.8,
}
RANDOM_STATE = 123


def create_user_item_matrix(
    interactions_df: pd.DataFrame,
    relevant_client_ids: np.ndarray,
    weighting_scheme: str = "custom_counts",
):
    """
    Creates a sparse user-item matrix with flexible weighting.

    Args:
        interactions_df: DataFrame with ['client_id', 'sku', 'interaction_type'].
        relevant_client_ids: Array of client IDs to include in the matrix rows.
        weighting_scheme: 'binary', 'counts', or 'custom_counts' (using INTERACTION_WEIGHTS).

    Returns:
        A tuple containing:
        - The sparse user-item matrix (CSR format).
        - client_id_map: Dictionary mapping client_id to matrix row index.
        - item_id_map: Dictionary mapping sku to matrix column index.
    """

    user_id_to_index = {client_id: i for i, client_id in enumerate(relevant_client_ids)}
    n_users = len(relevant_client_ids)

    # Filter interactions to only include relevant clients
    interactions_df = interactions_df[
        interactions_df["client_id"].isin(user_id_to_index)
    ]

    unique_items = interactions_df["sku"].unique()
    item_id_to_index = {sku: i for i, sku in enumerate(unique_items)}
    n_items = len(unique_items)

    # Calculate interaction values based on the chosen scheme
    if weighting_scheme == "binary":
        # Mark 1 for any interaction
        interactions_df["value"] = 1
        # Group by user/item and take max (or sum, but max ensures 1) in case of multiple interactions
        interaction_values = (
            interactions_df.groupby(["client_id", "sku"])["value"].max().reset_index()
        )
    elif weighting_scheme == "counts":
        # Simple count of interactions
        interaction_values = (
            interactions_df.groupby(["client_id", "sku"])
            .size()
            .reset_index(name="value")
        )
    elif weighting_scheme == "custom_counts":
        # Apply weights from INTERACTION_WEIGHTS
        interactions_df["value"] = (
            interactions_df["interaction_type"].map(INTERACTION_WEIGHTS).fillna(0)
        )
        interaction_values = (
            interactions_df.groupby(["client_id", "sku"])["value"].sum().reset_index()
        )
        # Optional: Clip negative values if desired (e.g., if remove cancels out add)
        # interaction_values['value'] = interaction_values['value'].clip(lower=0)
    else:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")

    # Filter out zero or negative interactions if they resulted from weighting
    interaction_values = interaction_values[interaction_values["value"] > 0]

    # Map client_ids and skus to their matrix indices
    row_indices = interaction_values["client_id"].map(user_id_to_index).values
    col_indices = interaction_values["sku"].map(item_id_to_index).values
    data_values = interaction_values["value"].values

    # Create the sparse matrix
    user_item_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)), shape=(n_users, n_items)
    )

    # --- Apply TF-IDF Transformation ---
    tfidf_transformer = TfidfTransformer(sublinear_tf=True, norm="l2")
    user_item_matrix_normalized = tfidf_transformer.fit_transform(user_item_matrix)

    return user_item_matrix_normalized, user_id_to_index, item_id_to_index


class SVDCalculator(Calculator):
    def __init__(self, embedding_dim: int, n_iter: int):
        self.embedding_dim = embedding_dim
        # Ensure embedding dim does not exceed challenge limits
        if self.embedding_dim > 2048:
            print(
                f"Warning: embedding_dim ({self.embedding_dim}) exceeds challenge limit (2048). Clamping."
            )
            self.embedding_dim = 2048
        self.n_iter = n_iter
        self.random_state = RANDOM_STATE

    @property
    def features_size(self) -> int:
        # not needed, maybe delete inheritance entirely
        return -1

    def compute_features(self, user_item_matrix: csr_matrix) -> np.ndarray:
        """Computes user embeddings using TruncatedSVD and converts to float16."""

        if user_item_matrix.shape[1] <= self.embedding_dim:
            print(
                f"Warning: Number of items ({user_item_matrix.shape[1]}) is less than or equal to embedding_dim ({self.embedding_dim}). Reducing embedding_dim."
            )
            effective_embedding_dim = max(
                1, user_item_matrix.shape[1] - 1
            )  # SVD requires n_components < n_features
        else:
            effective_embedding_dim = self.embedding_dim

        svd_model = TruncatedSVD(
            n_components=effective_embedding_dim,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        user_embeddings_float64 = svd_model.fit_transform(user_item_matrix)

        # If effective_embedding_dim was reduced, pad with zeros to reach the target embedding_dim
        if effective_embedding_dim < self.embedding_dim:
            print(
                f"Padding embeddings from {effective_embedding_dim} to {self.embedding_dim} dimensions."
            )
            padding = np.zeros(
                (
                    user_embeddings_float64.shape[0],
                    self.embedding_dim - effective_embedding_dim,
                )
            )
            user_embeddings_float64 = np.hstack((user_embeddings_float64, padding))

        user_embeddings_float16 = user_embeddings_float64.astype(np.float16)

        print(
            f"SVD explained variance ratio: {svd_model.explained_variance_ratio_.sum():.4f}"
        )
        print(f"Final embedding dtype: {user_embeddings_float16.dtype}")
        print(f"Final embedding shape: {user_embeddings_float16.shape}")

        return user_embeddings_float16
