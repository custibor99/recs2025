from tqdm import tqdm
import torch
from torch import nn


def train_matrix_factorization(model, dataloader, optimizer, device="cpu"):
    model.train()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        user_ids = batch["user"].to(device)
        pos_item_ids = batch["pos_item"].to(device)
        neg_item_ids = batch["neg_items"].to(device)

        

        optimizer.zero_grad()


        pos_scores = model(user_ids, pos_item_ids)
        if neg_item_ids.dim() == 2:
            _, n_repeat = neg_item_ids.shape
            user_ids = user_ids.repeat(n_repeat)
            neg_item_ids = neg_item_ids.flatten()
        neg_scores = model(user_ids, neg_item_ids)

        pos_scores = pos_scores.view(-1)
        neg_scores = neg_scores.view(-1)

        true_labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])
        predicted_scores = torch.cat([pos_scores, neg_scores])

        loss = bce_loss_fn(predicted_scores, true_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
