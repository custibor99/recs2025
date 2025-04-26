from __future__ import annotations

"""interaction_matrix_builder.py

Utility for turning raw interaction parquet files into a sparse client–product
interaction matrix *plus* explicit, well‑named mappings.

Mapping nomenclature
--------------------
We now adopt a clear **x_to_y** convention:

* **index_to_client_id** – NumPy array; position *i* → original ``client_id``
* **index_to_product_id** – NumPy array; position *j* → original ``product_id``
* **client_id_to_index** – ``dict`` generated lazily; original id → row index
* **product_id_to_index** – ``dict`` generated lazily; original id → col index

This should remove any ambiguity about mapping direction.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Tuple, Union

import numpy as np
import polars as pl
import scipy.sparse as sp

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


@dataclass(slots=True)
class InteractionDataset:
    """Bundle holding the matrix and both id - index mappings."""

    matrix: sp.csr_matrix
    index_to_client_id: np.ndarray  # idx - original client_id
    index_to_product_id: np.ndarray  # idx - original product_id

    def save(self, output_dir: Union[str, Path]) -> None:
        """Persist the sparse matrix and mapping arrays.

        ``interaction_matrix.npz``
            CSR matrix (SciPy)
        ``index_to_client_id.npy``
            NumPy array of original ``client_id`` ordered by row index
        ``index_to_product_id.npy``
            NumPy array of original ``product_id`` ordered by column index
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sp.save_npz(output_dir / "interaction_matrix.npz", self.matrix)
        np.save(output_dir / "index_to_client_id.npy", self.index_to_client_id)
        np.save(output_dir / "index_to_product_id.npy", self.index_to_product_id)

        logger.info("Dataset saved to %s", output_dir)

    @classmethod
    def load(cls, output_dir: Union[str, Path]) -> "InteractionDataset":
        """Load an :class:`InteractionDataset` that was saved with :pymeth:`save`."""
        output_dir = Path(output_dir)
        matrix = sp.load_npz(output_dir / "interaction_matrix.npz")
        index_to_client_id = np.load(output_dir / "index_to_client_id.npy")
        index_to_product_id = np.load(output_dir / "index_to_product_id.npy")
        logger.info("Dataset loaded from %s", output_dir)
        return cls(matrix, index_to_client_id, index_to_product_id)

    # ------------------------------
    # Mapping helpers
    # ------------------------------
    def client_id_to_index(self, client_id: Union[int, str]) -> int:
        """Original ``client_id`` -> contiguous row index."""
        if not hasattr(self, "_client_id_to_index"):
            self._client_id_to_index = {
                cid: i for i, cid in enumerate(self.index_to_client_id)
            }
        return self._client_id_to_index[client_id]

    def product_id_to_index(self, product_id: Union[int, str]) -> int:
        """Original ``product_id`` -> contiguous column index."""
        if not hasattr(self, "_product_id_to_index"):
            self._product_id_to_index = {
                pid: j for j, pid in enumerate(self.index_to_product_id)
            }
        return self._product_id_to_index[product_id]

    def index_to_client_id(self, idx: int):
        """Row index -> original ``client_id``."""
        return self.index_to_client_id[idx]

    def index_to_product_id(self, idx: int):
        """Column index -> original ``product_id``."""
        return self.index_to_product_id[idx]


# ---------------------------------------------------------------------------
# Core build helper functions
# ---------------------------------------------------------------------------


def _load_single_category(path: Union[str, Path], effect: int) -> pl.DataFrame:
    """Load a parquet, drop *timestamp*, add constant *effect* column."""
    logger.debug("Reading %s", path)
    df = pl.read_parquet(path, low_memory=True)
    if "timestamp" in df.columns:
        df = df.drop("timestamp")
    df = df.with_columns(pl.lit(effect).alias("effect"))
    df = df.rename({"sku": "product_id"})
    return df


def _build_mappings(df: pl.DataFrame) -> Tuple[pl.DataFrame, np.ndarray, np.ndarray]:
    """Turn arbitrary ids into contiguous indices and deliver mapping arrays."""
    index_to_client_id = (
        df.select("client_id").unique().sort("client_id").to_numpy().flatten()
    )
    index_to_product_id = (
        df.select("product_id").unique().sort("product_id").to_numpy().flatten()
    )

    client_id_to_index = {cid: i for i, cid in enumerate(index_to_client_id)}
    product_id_to_index = {pid: j for j, pid in enumerate(index_to_product_id)}

    df = df.with_columns(
        pl.col("client_id").replace(client_id_to_index).alias("client_id"),
        pl.col("product_id").replace(product_id_to_index).alias("product_id"),
    )
    return df, index_to_client_id, index_to_product_id


def _to_sparse(
    df: pl.DataFrame,
    n_clients: int,
    n_products: int,
    dtype=np.int16,
) -> sp.csr_matrix:
    """Convert the aggregated frame into a CSR matrix."""
    row = df["client_id"].to_numpy()
    col = df["product_id"].to_numpy()
    data = df["effect"].to_numpy()

    coo = sp.coo_matrix((data, (row, col)), shape=(n_clients, n_products), dtype=dtype)
    return coo.tocsr()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
CategorySpec = Mapping[str, Union[str, int]]


def build_dataset(
    data_categories: Mapping[str, CategorySpec],
    *,
    dtype=np.int16,
) -> InteractionDataset:
    """Build an :class:`InteractionDataset` from *data_categories*.

    Parameters
    ----------
    data_categories
        ``{"name": {"file": "...", "effect": +/- int}, ...}``
    dtype
        Numpy dtype for the sparse matrix (default ``np.int16``).
    """

    frames: List[pl.DataFrame] = []
    for name, spec in data_categories.items():
        try:
            frames.append(_load_single_category(spec["file"], int(spec["effect"])))
        except Exception:
            logger.exception("Failed loading %s from %s", name, spec.get("file"))
            raise

    interactions = pl.concat(frames)
    interactions = interactions.group_by(
        ["client_id", "product_id"], maintain_order=False
    ).agg(pl.sum("effect").alias("effect"))

    # Drop pairs whose net effect is zero – they contribute nothing.
    # interactions = interactions.filter(pl.col("effect") != 0)

    interactions, idx_to_cli, idx_to_prod = _build_mappings(interactions)

    csr = _to_sparse(interactions, len(idx_to_cli), len(idx_to_prod), dtype=dtype)

    return InteractionDataset(csr, idx_to_cli, idx_to_prod)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import textwrap

    parser = argparse.ArgumentParser(
        description="Build an interaction matrix from parquet files.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """Example usage:
              python interaction_matrix_builder.py \\
                --spec spec.json \\
                --output cdae/data/output
            ``spec.json`` could look like:
              {
                "buy":    {"file": "data/input/product_buy.parquet",   "effect": 2},
                "add":    {"file": "data/input/add_to_cart.parquet",   "effect": 1},
                "remove": {"file": "data/input/remove_from_cart.parquet", "effect": -1}
              }
            """,
        ),
    )
    parser.add_argument("--spec", help="Path to JSON file with category spec")
    parser.add_argument("--output", required=True, help="Where to save the dataset")
    args = parser.parse_args()

    if args.spec:
        spec_path = Path(args.spec)
        if not spec_path.exists():
            parser.error(f"Spec file {spec_path} does not exist")

        category_spec = json.loads(spec_path.read_text())
    else:
        category_spec = {
            "buy": {"file": "data/input/product_buy.parquet", "effect": 2},
            "add": {"file": "data/input/add_to_cart.parquet", "effect": 1},
            "remove": {"file": "data/input/remove_from_cart.parquet", "effect": -1},
        }
    ds = build_dataset(category_spec)
    ds.save(args.output)
    print("✅ Dataset built and saved\n")
