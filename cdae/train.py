from data_utils.interaction_dataset import InteractionDataset, InteractionData
from torch.utils.data import DataLoader
import torch

from cdae.model.CDAE import CDAE
from cdae.model.trainer import CDAETrainer

import argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory with input and target data - produced by data_utils.interaction_dataset",
    )
    parser.add_argument(
        "--model-save-dir",
        type=str,
        required=True,
        help="Directory where to save the trained model",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs to train the model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adagrad", "rmsprop"], 
        help="Optimizer to use for training",
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    params = parser.parse_args()

    DATA_DIR = params.data_dir
    MODEL_SAVE_DIR = params.model_save_dir
    NUM_EPOCHS = params.num_epochs
    BATCH_SIZE = params.batch_size
    LEARNING_RATE = params.learning_rate
    OPTIMIZER = params.optimizer

    print(f"Loading dataset from {DATA_DIR}")

    ds = InteractionData.load("data")
    torch_ds = InteractionDataset(ds, to_dense=True)
    print(
        f"Loaded dataset with {len(torch_ds)} clients and {torch_ds._matrix.shape[1]} products\n"
    )

    num_users, num_items = len(torch_ds), torch_ds._matrix.shape[1]

    row, client_id = torch_ds[1]

    model = CDAE(
        num_users=num_users,
        num_items=num_items,
        hidden_dim=64,
        index_to_uid=ds.index_to_client_id,
        index_to_iid=ds.index_to_product_id,
        uid_to_index=ds.client_id_to_index,
        iid_to_index=ds.product_id_to_index,
    )

    trainer = CDAETrainer(
        model,
        lr=LEARNING_RATE,
        optimizer_method=OPTIMIZER,
        device= "cuda" if torch.cuda.is_available() else "cpu",
    )

    dl = DataLoader(
        torch_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    trainer.fit(
        dataloader=dl,
        num_epochs=NUM_EPOCHS,
    )
    
    print(f"Saving model to {MODEL_SAVE_DIR}")
    torch.save(model.state_dict(), MODEL_SAVE_DIR)
    print("Model saved successfully.")
    print("Training completed.")

