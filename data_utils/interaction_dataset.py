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
from typing import List, Mapping, Tuple, Union, Optional

import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class InteractionDataset(Dataset):
    """
    PyTorch ``Dataset`` wrapper around :class:`InteractionData`.

    Each iteration yields a **single client's interaction vector** together with
    its *original* ``client_id``.

    Example
    -------
    >>> from interaction_dataset import InteractionData, InteractionDataset
    >>> ds = InteractionData.load("path/to/built_dataset")
    >>> torch_ds = InteractionDataset(ds, to_dense=True)
    >>> row, client_id = torch_ds[0]
    >>> row.shape, client_id
    (torch.Size([n_products]), 12345)

    The dataset supports both dense and sparse PyTorch tensors. For sparse output,
    set ``to_dense=False`` (default).
    """

    def __init__(
        self,
        interaction_dataset: InteractionData,
        *,
        to_dense: bool = True,
        dtype: torch.dtype = torch.int16,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        """PyTorch ``Dataset`` returning [row_tensor, client_id].

        Parameters
        ----------
        interaction_dataset
            Pre-built :class:`InteractionData` (can be loaded from disk).
        to_dense
            If *True* (default), convert the CSR row to a **dense** 1-D ``torch.Tensor``.
            If *False*, return a **sparse** ``torch.Tensor`` in COO format.
        dtype
            ``torch.dtype`` for the interaction values (default ``torch.int16``).
        device
            Optional device to place tensors on. If *None*, keep them on CPU.
        """
        self._ds = interaction_dataset
        self._matrix = interaction_dataset.matrix  # SciPy CSR
        self._client_ids = interaction_dataset.index_to_client_id  # NumPy array
        self._to_dense = to_dense
        self._dtype = dtype
        self._device = device if device is not None else "cpu"

    # ---------------------------------------------------------------------
    # PyTorch Dataset API
    # ---------------------------------------------------------------------
    def __len__(self) -> int:
        return self._matrix.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[int, str]]:
        """Return *(interaction_row, client_id)* for the given *idx*."""

        row_csr = self._matrix.getrow(idx)  # (1 x n_products)

        if self._to_dense:
            # Convert to a dense 1‑D tensor
            tensor = torch.as_tensor(
                row_csr.toarray(), dtype=self._dtype, device=self._device
            ).squeeze(0)
        else:
            raise NotImplementedError(
                "Sparse COO tensor construction not implemented yet. "
            )

        client_id = self._client_ids[idx]
        return tensor, client_id


@dataclass(slots=True)
class InteractionData:
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
    def load(cls, output_dir: Union[str, Path]) -> "InteractionData":
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
) -> InteractionData:
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

    return InteractionData(csr, idx_to_cli, idx_to_prod)


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
              python interaction_dataset.py \\
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

    # Example usage
    ds = InteractionData.load(args.output)
    torch_ds = InteractionDataset(ds, to_dense=True)
    print(
        f"Loaded dataset with {len(torch_ds)} clients and {torch_ds._matrix.shape[1]} products\n"
    )

    row, client_id = torch_ds[1]
    print(f"Row shape: {row.shape}, Client ID: {client_id}")
    print("✅ Example usage complete\n")

    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    dl = DataLoader(
        torch_ds,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    for batch in tqdm(dl):
        row, client_id = batch
        print(f"Batch shape: {row.shape}, Client IDs: {client_id}")
        print(f"Batch Memory: {row.element_size() * row.nelement() / (1024**2):.2f} MB")

    print("✅ DataLoader example complete\n")
