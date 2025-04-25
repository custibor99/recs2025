import logging
from typing import Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from baseline.aggregated_features_baseline.constants import EventTypes
from baseline.aggregated_features_baseline.create_embeddings import (
    load_relevant_clients_ids,
    save_embeddings,
    get_parser,
)
from data_utils.utils import (
    load_with_properties,
)
from data_utils.data_dir import DataDir
from leon.SVDCalculator import create_user_item_matrix, SVDCalculator

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


EMBEDDING_DIM = 128
SVD_ITERATIONS = 15


def create_embeddings_svd(
    data_dir: "DataDir",
    relevant_client_ids: np.ndarray,
    weighting_scheme: str = "custom_counts",  # Options: 'binary', 'counts', 'custom_counts'
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates user embeddings using SVD on an enhanced user-item interaction matrix."""

    logger.info("Loading data...")
    interactions_df = _load_and_process_data(data_dir)

    logger.info(
        f"Creating user item matrix using weighting scheme: {weighting_scheme}..."
    )
    user_item_matrix, client_id_map, item_id_map = create_user_item_matrix(
        interactions_df, relevant_client_ids, weighting_scheme=weighting_scheme
    )

    logger.info(
        f"Generating embeddings using SVD with n_components={EMBEDDING_DIM} and n_iter={SVD_ITERATIONS}..."
    )
    svd_calculator = SVDCalculator(embedding_dim=EMBEDDING_DIM, n_iter=SVD_ITERATIONS)
    embeddings = svd_calculator.compute_features(user_item_matrix)

    return relevant_client_ids, embeddings


def _load_and_process_data(data_dir: "DataDir") -> pd.DataFrame:
    product_buy = load_with_properties(
        data_dir=data_dir, event_type=EventTypes.PRODUCT_BUY.value
    )
    product_buy["interaction_type"] = EventTypes.PRODUCT_BUY.value

    add_to_cart = load_with_properties(
        data_dir=data_dir, event_type=EventTypes.ADD_TO_CART.value
    )
    add_to_cart["interaction_type"] = EventTypes.ADD_TO_CART.value

    remove_from_cart = load_with_properties(
        data_dir=data_dir, event_type=EventTypes.REMOVE_FROM_CART.value
    )
    remove_from_cart["interaction_type"] = EventTypes.REMOVE_FROM_CART.value

    interactions_for_matrix = pd.concat(
        [product_buy, add_to_cart, remove_from_cart],
        ignore_index=True,
    )[["client_id", "sku", "interaction_type", "timestamp"]]

    return interactions_for_matrix


def main(params):
    data_dir = DataDir(Path(params.data_dir))

    embeddings_dir = Path(params.embeddings_dir)

    relevant_client_ids = load_relevant_clients_ids(input_dir=data_dir.input_dir)
    client_ids, embeddings = create_embeddings_svd(
        data_dir=data_dir,
        relevant_client_ids=relevant_client_ids,
    )

    save_embeddings(
        client_ids=client_ids,
        embeddings=embeddings,
        embeddings_dir=embeddings_dir,
    )


if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()
    main(params=params)
