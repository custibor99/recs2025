import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Callable

from cdae.model.utils import _get_activation


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
        num_users: List[int],
        num_items: List[int],
        hidden_dim: int = 64,
        *,
        index_to_uid: Callable,
        index_to_iid: Callable,
        uid_to_index: Callable,
        iid_to_index: Callable,
        f_act: str = "sigmoid",
        g_act: str = "sigmoid",
        corruption_level: float = 0.3,
    ) -> None:
        super().__init__()

        # self.user_ids = user_ids
        # self.item_ids = item_ids
        self.num_users = num_users
        self.num_items = num_items

        self.index_to_uid = index_to_uid
        self.index_to_iid = index_to_iid
        self.uid_to_index = uid_to_index
        self.iid_to_index = iid_to_index

        self.hidden_dim = hidden_dim
        self.corruption_level = corruption_level

        # encoder + decoder
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        # user-specific embedding for hidden layer
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)

        # activations
        self.f_act = _get_activation(f_act)
        self.g_act = _get_activation(g_act)

    def forward(
        self,
        R: torch.Tensor,  # (B, num_items)
        user_id: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        
        # 1. corrupt input
        R_tilde = F.dropout(R, p=self.corruption_level, training=self.training)

        # 2. encode
        h = R_tilde
        h = self.f_act(self.encoder(h))  # (B, hidden_dim)

        # 3. user‑specific embedding
        user_idx = torch.tensor(
            [self.uid_to_index(int(uid)) for uid in user_id], device=user_id.device
        )
        h = h + self.user_embedding(user_idx)

        # 4. decode
        x = self.g_act(self.decoder(h))  # (B, num_items)

        # 5. output to get probs
        x = torch.sigmoid(x)  # (B, num_items)
        return h, x