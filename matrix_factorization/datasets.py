import torch
from torch.utils.data import Dataset
import random


class UserItemInteractionDataset(Dataset):
    def __init__(self, df, user_map:dict[int,int], item_map:dict[int,int], category_map:dict[int,int], num_negatives=1):
        """
        df: DataFrame with 'client_id' and 'sku' columns
        num_negatives: Number of negative samples per user per positive
        """
        df = df.drop_duplicates(subset=["client_id", "sku"])

        self.user_map = user_map
        self.item_map = item_map
        self.num_users = len(self.user_map)
        self.num_items = len(self.item_map)
        self.num_negatives = num_negatives
        # Encode to categorical indices
        df["user_idx"] = df["client_id"].astype("category").cat.codes
        df["item_idx"] = df["sku"].astype("category").cat.codes

        self.positive_pairs = list(zip(df["user_idx"], df["item_idx"]))

        # Build user -> positive item set for negative sampling
        self.user_pos_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    def __len__(self):
        return len(self.positive_pairs)

    def __getitem__(self, idx):
        # Check if idx is within bounds
        if idx < 0 or idx >= len(self.positive_pairs):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.positive_pairs)}.")

        user, pos_item = self.positive_pairs[idx]

        # Sample negatives
        neg_items = []
        attempts = 0
        max_attempts = 10 * self.num_negatives  # Arbitrary limit to avoid infinite loop
        
        while len(neg_items) < self.num_negatives and attempts < max_attempts:
            neg = random.randint(0, self.num_items - 1)
            if neg not in self.user_pos_items[user] and neg != pos_item:
                neg_items.append(neg)
            attempts += 1
        
        # Return all the found negative items, even if it's less than num_negatives
        return {
            "user": torch.tensor(user, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_items": torch.tensor(neg_items, dtype=torch.long),
        }