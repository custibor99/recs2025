"""
cdae_inference.py
-----------------
Inference pipeline for a trained CDAE recommender.
Usage (single user):
    python cdae_inference.py \
        --data-dir path/to/data \
        --model-checkpoint path/to/cdae_state_dict.pt \
        --hidden-dim 64 \
        --user-id 123456 \
        --top-k 20
Usage (all users to CSV):
    python cdae_inference.py \
        --data-dir path/to/data \
        --model-checkpoint path/to/cdae_state_dict.pt \
        --hidden-dim 64 \
        --top-k 50 \
        --output-csv recs.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from warnings import warn

import numpy as np
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset

from cdae.model.CDAE import CDAE
from data_utils.interaction_dataset import InteractionData, InteractionDataset


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, type=str,
                   help="Directory that contains the pre-processed InteractionData")
    p.add_argument("--model-checkpoint", required=True, type=str,
                   help="Path to *.pt file saved with torch.save(model.state_dict(), …)")
    p.add_argument("--hidden-dim", required=True, type=int,
                   help="Must match the one used at training time")
    p.add_argument("--relevant-clients", required=True, type=str,
                   help="Path to a .npy file with a list of client IDs to use for inference")
    p.add_argument("--output-dir", required=True, type=str,
                     help="Directory to save the output files")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = get_parser().parse_args()

    # -------------------------------------------------------------------------
    # Load data & mappings
    # -------------------------------------------------------------------------
    ds: InteractionData = InteractionData.load(args.data_dir)
    torch_ds = InteractionDataset(ds)
    num_users = ds.matrix.shape[0]
    num_items = ds.matrix.shape[1]

    # -------------------------------------------------------------------------
    # Build model skeleton and load weights
    # -------------------------------------------------------------------------
    model = CDAE(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=args.hidden_dim,
        index_to_uid=ds.index_to_client_id,
        index_to_iid=ds.index_to_product_id,
        uid_to_index=ds.client_id_to_index,
        iid_to_index=ds.product_id_to_index,
    )
    model.load(args.model_checkpoint)
    model.to(args.device)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    relevant_clients = np.load(args.relevant_clients)

    valid_clients = relevant_clients[np.isin(relevant_clients, ds._index_to_client_id)]
    valid_client_idxs = np.array([ds.client_id_to_index(client_id) for client_id in valid_clients])
    
    print(f"Number of valid clients: {len(valid_clients)}")
    print(f"Number of total clients: {len(relevant_clients)}")

    subset = Subset(torch_ds, valid_client_idxs)

    loader = DataLoader(
        dataset=subset,
        batch_size=10,
        num_workers=0,
    )


    emb_out_file = os.path.join(args.output_dir, "embeddings.npy")
    embeddings = np.lib.format.open_memmap( # Write to file without loading into memory
        emb_out_file,
        mode='w+',
        dtype=np.float16,
        shape=(len(valid_clients), args.hidden_dim),
    )

    with torch.no_grad():
        model.eval()
        for i, (R, batch_ids) in enumerate(tqdm(loader)):
            R, batch_ids = R.to(torch.float32).to(args.device), batch_ids.to(args.device)
            embedding, _ = model(R, batch_ids)
            embeddings[i * loader.batch_size:(i + 1) * loader.batch_size] = embedding.cpu().numpy()
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")

    # -------------------------------------------------------------------------
    # Save to client_ids.npy
    # -------------------------------------------------------------------------
    client_ids = np.array(valid_clients)
    np.save(args.output_dir + "/client_ids.npy", client_ids)
    print("Saved client_ids.npy and embeddings.npy")



if __name__ == "__main__":
    main()
