from sklearn.metrics import roc_auc_score  # Importing AUC from sklearn
import torch
import numpy as np


def evaluate(model, dataloader, device):
    model.eval()
    all_true_labels = []
    all_pred_scores = []

    with torch.no_grad():
        for batch in dataloader:
            user_ids = batch["user"].to(device)
            pos_item_ids = batch["pos_item"].to(device)
            neg_item_ids = batch["neg_items"].to(device)

            # Get positive item scores
            pos_scores = model(user_ids, pos_item_ids)

            if neg_item_ids.dim() == 2:
                _, n_repeat = neg_item_ids.shape
                user_ids = user_ids.repeat(n_repeat)
                neg_item_ids = neg_item_ids.flatten()
            # Get negative item scores
            neg_scores = model(user_ids, neg_item_ids)

            # Ensure both pos_scores and neg_scores are 1D tensors
            pos_scores = pos_scores.view(-1)  # Flatten to 1D if necessary
            neg_scores = neg_scores.view(-1)  # Flatten to 1D if necessary

            # Create true labels: 1 for positive items, 0 for negative items
            true_labels = torch.cat(
                [torch.ones_like(pos_scores), torch.zeros_like(neg_scores)]
            )
            predicted_scores = torch.cat([pos_scores, neg_scores])

            # Collect true ratings and predicted scores
            all_true_labels.extend(true_labels.cpu().numpy())
            all_pred_scores.extend(predicted_scores.cpu().numpy())

    # Calculate MSE
    mse = ((np.array(all_true_labels) - np.array(all_pred_scores)) ** 2).mean()

    # Calculate AUC using sklearn's roc_auc_score
    auc = roc_auc_score(all_true_labels, all_pred_scores)
    return mse, auc
