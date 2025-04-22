import numpy as np
import logging
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


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
    temporal_decay: float = 0.9,  # Decay factor for temporal weighting
):
    user_id_to_index = {client_id: i for i, client_id in enumerate(relevant_client_ids)}
    n_users = len(relevant_client_ids)

    # Filter interactions to only include relevant clients
    interactions_df = interactions_df[
        interactions_df["client_id"].isin(user_id_to_index)
    ].copy()  # copy to avoid warning
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])

    # Calculate recency weights
    max_timestamp = interactions_df["timestamp"].max()
    # Convert time difference to days
    interactions_df["days_old"] = (
        max_timestamp - interactions_df["timestamp"]
    ).dt.total_seconds() / (24 * 3600)
    # Apply exponential decay
    interactions_df["recency_weight"] = temporal_decay ** interactions_df["days_old"]

    # Get unique items
    unique_items = interactions_df["sku"].unique()
    item_id_to_index = {sku: i for i, sku in enumerate(unique_items)}
    n_items = len(unique_items)

    # Calculate interaction values based on the chosen scheme
    if weighting_scheme == "binary":
        # Mark 1 for any interaction, weighted by recency
        interactions_df["value"] = interactions_df["recency_weight"]
        # Group by user/item and take max in case of multiple interactions
        interaction_values = (
            interactions_df.groupby(["client_id", "sku"])["value"].max().reset_index()
        )
    elif weighting_scheme == "counts":
        # Weight each interaction by recency and sum the weighted counts
        interactions_df["value"] = interactions_df["recency_weight"]
        interaction_values = (
            interactions_df.groupby(["client_id", "sku"])["value"].sum().reset_index()
        )
    elif weighting_scheme == "custom_counts":
        # Apply weights from INTERACTION_WEIGHTS and multiply by recency weight
        interactions_df["value"] = (
            interactions_df["interaction_type"].map(INTERACTION_WEIGHTS).fillna(0)
            * interactions_df["recency_weight"]
        )
        interaction_values = (
            interactions_df.groupby(["client_id", "sku"])["value"].sum().reset_index()
        )
    else:
        raise ValueError(f"Unknown weighting_scheme: {weighting_scheme}")

    # Filter out zero or negative interactions
    interaction_values = interaction_values[interaction_values["value"] > 0]

    # Create the sparse matrix
    row_indices = interaction_values["client_id"].map(user_id_to_index).values
    col_indices = interaction_values["sku"].map(item_id_to_index).values
    data_values = interaction_values["value"].values

    user_item_matrix = csr_matrix(
        (data_values, (row_indices, col_indices)), shape=(n_users, n_items)
    )

    # Apply TF-IDF Transformation
    tfidf_transformer = TfidfTransformer(sublinear_tf=True, norm="l2")
    user_item_matrix_normalized = tfidf_transformer.fit_transform(user_item_matrix)

    return user_item_matrix_normalized, user_id_to_index, item_id_to_index


def create_page_visit_matrix(
    page_visit_df: pd.DataFrame, relevant_client_ids
) -> csr_matrix:
    page_visit_df = page_visit_df[page_visit_df["client_id"].isin(relevant_client_ids)]
    user_id_to_index = {client_id: i for i, client_id in enumerate(relevant_client_ids)}
    n_users = len(relevant_client_ids)

    # Create user-URL matrix
    unique_urls = page_visit_df["url"].unique()
    url_to_index = {url: i for i, url in enumerate(unique_urls)}

    # Count URL visits
    url_counts = (
        page_visit_df.groupby(["client_id", "url"]).size().reset_index(name="count")
    )

    # Map to row and column indices
    url_row_indices = [
        user_id_to_index[cid]
        for cid in url_counts["client_id"]
        if cid in user_id_to_index
    ]
    url_col_indices = [url_to_index[url] for url in url_counts["url"]]

    # Create sparse matrix
    page_matrix = csr_matrix(
        (
            url_counts["count"].values[: len(url_row_indices)],
            (url_row_indices, url_col_indices[: len(url_row_indices)]),
        ),
        shape=(n_users, len(unique_urls)),
    )

    # Apply TF-IDF
    tfidf_transformer = TfidfTransformer(sublinear_tf=True, norm="l2")
    page_matrix = tfidf_transformer.fit_transform(page_matrix)

    return page_matrix


class SVDCalculator:
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

        user_embeddings_float64 = normalize(
            user_embeddings_float64
        )  # Normalize to reduce bias for more frequent words

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
