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
from svd.SVDCalculator import (
    create_user_item_matrix,
    SVDCalculator,
    create_page_visit_matrix,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


EMBEDDING_DIM = 256
SVD_ITERATIONS = 10


def create_embeddings_svd(
    data_dir: "DataDir",
    relevant_client_ids: np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
    product_weight: float = 0.6,
    page_weight: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Loading all data types...")
    interactions_df = _load_and_process_data(data_dir)

    logger.info("Creating product interaction embeddings...")
    product_matrix, user_id_to_index, _ = create_user_item_matrix(
        interactions_df,
        relevant_client_ids,
        weighting_scheme="custom_counts",
        temporal_decay=0.9,
    )
    product_dim = int(embedding_dim * product_weight)
    svd_calculator = SVDCalculator(embedding_dim=product_dim, n_iter=SVD_ITERATIONS)
    product_embeddings = svd_calculator.compute_features(product_matrix)
    logger.info(f"Created product embeddings with shape: {product_embeddings.shape}")

    logger.info("Creating page visit embeddings...")
    page_visit_df = load_with_properties(
        data_dir=data_dir, event_type=EventTypes.PAGE_VISIT.value
    )
    page_matrix = create_page_visit_matrix(page_visit_df, relevant_client_ids)
    page_dim = int(embedding_dim * page_weight)
    page_svd = SVDCalculator(embedding_dim=page_dim, n_iter=SVD_ITERATIONS)
    page_embeddings = page_svd.compute_features(page_matrix)
    logger.info(f"Created page visit embeddings with shape: {page_embeddings.shape}")

    combined_embeddings = np.hstack([product_embeddings, page_embeddings])

    return relevant_client_ids, combined_embeddings


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
