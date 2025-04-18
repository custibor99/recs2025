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

from leon.SVDCalculator import SVDCalculator, create_user_item_matrix

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

EMBEDDING_DIM = 512


def create_embeddings_svd(
    data_dir: DataDir, relevant_client_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Loading data...")
    interactions_df = _load_interactions(data_dir)

    logger.info("Create user item matrix")
    user_item_matrix = create_user_item_matrix(interactions_df, relevant_client_ids)

    logger.info("Generate embeddings using SVD")
    svd_calculator = SVDCalculator(EMBEDDING_DIM)
    embeddings = svd_calculator.compute_features(user_item_matrix)

    return relevant_client_ids, embeddings


def _load_interactions(data_dir: DataDir) -> pd.DataFrame:
    product_buy = load_with_properties(
        data_dir=data_dir, event_type=EventTypes.PRODUCT_BUY.value
    )
    product_buy["timestamp"] = pd.to_datetime(product_buy.timestamp)

    add_to_cart = load_with_properties(
        data_dir=data_dir, event_type=EventTypes.ADD_TO_CART.value
    )
    add_to_cart["timestamp"] = pd.to_datetime(add_to_cart.timestamp)

    interactions_df = pd.concat(
        [product_buy[["client_id", "sku"]], add_to_cart[["client_id", "sku"]]],
        ignore_index=True,
    )

    return interactions_df


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
