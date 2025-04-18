import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional

from utils import _get_activation, _get_optimizer


class CDAE(nn.Module):
    """Collaborative Denoising Auto-Encoder.
    This class implements a Collaborative Denoising Auto-Encoder (CDAE) layer, a type of
    neural network used for unsupervised learning. The CDAE is designed to learn a
    representation of the input data by reconstructing it from a corrupted version.
    The CDAE consists of an encoder and a decoder, both implemented as linear layers.
    The CDAE is a variant of the Denoising Auto-Encoder (DAE) that incorporates
    collaborative filtering techniques to improve the performance of the model.

    This implementation is taken from the original CDAE paper but adapted from
    TensorFlow to PyTorch. The original code can be found here:

    https://github.com/gtshs2/Collaborative-Denoising-Auto-Encoder/blob/master/src/CDAE.py
    https://github.com/yoongi0428/RecSys_PyTorch/blob/master/models/CDAE.py
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        hidden_dim: int = 64,
        *,
        f_act: str = "sigmoid",
        g_act: str = "sigmoid",
        corruption_level: float = 0.3,
    ) -> None:
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.hidden_dim = hidden_dim
        self.corruption_level = corruption_level

        # encoder + decoder
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        # user-specific embedding for first hidden layer
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)

        # activations
        self.f_act = _get_activation(f_act)
        self.g_act = _get_activation(g_act)

    def forward(
        self,
        R: torch.Tensor,  # (B, num_items)
        user_idx: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        # 1. corrupt input
        R_tilde = F.dropout(R, p=self.corruption_level, training=self.training)

        # 2. encode
        h = R_tilde
        h = self.f_act(self.encoder(h))  # (B, hidden_dim)

        # 3. user‑specific embedding
        h = h + self.user_embedding(user_idx)

        # 4. decode
        x = self.g_act(self.decoder(h))  # (B, num_items)

        # 5. output to get probs
        x = torch.sigmoid(x)  # (B, num_items)
        return x


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

    def fit_epoch(  # -> fit single epoch
        self,
        self_R: torch.Tensor,
        *,
        batch_size: int,
        shuffle: bool = True,
    ) -> float:
        self.model.train()
        dataset = TensorDataset(self_R, torch.arange(self_R.size(0)))  # -> user_idx
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        epoch_loss = 0.0
        for R, usr in loader:
            R = R.to(self.device)
            usr = usr.to(self.device)

            self.opt.zero_grad()

            decoded = self.model(R, usr)

            loss = F.binary_cross_entropy(decoded, R, reduction="none").sum(1).mean()
            loss.backward()

            self.opt.step()
            epoch_loss += loss.item() * R.size(0)

        return epoch_loss / len(self_R)

    def fit(
        self,
        self_R: torch.Tensor,
        *,
        batch_size: int,
        num_epochs: int,
        shuffle: bool = True,
    ) -> None:
        for ep in range(num_epochs):
            epoch_loss = self.fit_epoch(self_R, batch_size=batch_size, shuffle=shuffle)
            print(f"Epoch {ep + 1} | Train loss {epoch_loss:.4f}")

    @torch.no_grad()  # no dropout, no corruption
    def predict(self, R: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        user_idx = torch.arange(R.size(0), device=self.device)
        return self.model(R.to(self.device), user_idx).cpu()


# --- EXAMPLE USAGE ---

num_users, num_items = 1000, 10_000
R_np = (np.random.rand(num_users, num_items) < 0.05).astype(
    np.float32
)  # 5% got ratings

cdae = CDAE(
    num_users=num_users,
    num_items=num_items,
    hidden_dim=64,
    f_act="relu",
    g_act="relu",
    corruption_level=0.2,
)

trainer = CDAETrainer(cdae, lr=1e-3)

R = torch.tensor(R_np)

trainer.fit(
    self_R=R,
    batch_size=256,
    num_epochs=12,
    shuffle=True,
)

R_pred = trainer.predict(R)
print(f"Prediction: {R_pred.shape}")  # (num_users, num_items)
