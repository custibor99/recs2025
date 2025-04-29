from cdae.model.CDAE import CDAE
from cdae.model.utils import _get_optimizer
from typing import Optional
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os

class CDAETrainer:
    def __init__(
        self,
        model: CDAE,
        *,
        lr: float = 1e-4,
        optimizer_method: str = "Adam",
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model.to(self.device)

        self.opt = _get_optimizer(optimizer_method)(self.model.parameters(), lr=lr)
        
    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        num_epochs: int,
        checkpoint_path: Optional[str] = None,
        log_fn: Optional[callable] = None,
    ) -> None:
        self.model.train()
        for ep in tqdm(range(num_epochs), total=num_epochs, unit="epoch", desc="Training"):
            epoch_loss = 0.0
            for R, user_ids in tqdm(dataloader, desc=f"Epoch {ep+1}/{num_epochs}", total=len(dataloader), unit="batch"):
                # PyTorch requires float tensors
                # convert to float32
                if R.dtype != torch.float32:
                    R = R.float()
                R = R.to(self.device)
                user_ids = user_ids.to(self.device)

                self.opt.zero_grad()

                _, decoded = self.model(R, user_ids)

                loss = nn.MSELoss()(decoded, R)
                loss.backward()

                self.opt.step()
                epoch_loss += loss.item() * R.size(0)
            
            if checkpoint_path:
                self.model.save(os.path.join(checkpoint_path, f"epoch_{ep+1}.pth"))
            
            if log_fn:
                log_fn(ep, {"loss": epoch_loss / len(dataloader.dataset)})
            print(f"Epoch {ep}, Average Loss: {epoch_loss / len(dataloader.dataset)}")

# --- EXAMPLE USAGE ---

if __name__ == "__main__":
    
    user_ids = np.linspace(10, 10000, 1000).astype(np.int64)
    item_ids = np.linspace(1, 1000, 1000).astype(np.int64)
    
    index_to_uid = lambda i: user_ids[i]
    index_to_iid = lambda i: item_ids[i]
    uid_to_index = lambda uid: np.where(user_ids == uid)[0][0]
    iid_to_index = lambda iid: np.where(item_ids == iid)[0][0]

    num_users, num_items = len(user_ids), len(item_ids)
    R_np = (np.random.rand(num_users, num_items) < 0.05).astype(
        np.float32
    )  # 5% got ratings
    
    dataloader = DataLoader(
        TensorDataset(
            torch.tensor(R_np),
            torch.tensor(user_ids),
        ),
        batch_size=8,
        shuffle=True,
    )
    model = CDAE(num_users=num_users,
                 num_items=num_items,
                 hidden_dim=64,
                 index_to_uid=index_to_uid,
                 index_to_iid=index_to_iid,
                 uid_to_index=uid_to_index,
                 iid_to_index=iid_to_index)
    
    trainer = CDAETrainer(model, lr=1e-4, optimizer_method="Adam")
    trainer.fit(dataloader, num_epochs=10, checkpoint_path=".")
    
    model = CDAE(num_users=num_users,
                 num_items=num_items,
                 hidden_dim=64,
                 index_to_uid=index_to_uid,
                 index_to_iid=index_to_iid,
                 uid_to_index=uid_to_index,
                 iid_to_index=iid_to_index)
    
    model.load("epoch_10.pth")
    
    print("Model loaded successfully.")
    print("Model parameters...")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}")