import torch.nn as nn
import torch


class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(MatrixFactorization, self).__init__()

        # Latent factor matrices for users and items
        self.user_matrix = nn.Embedding(num_users, latent_dim)
        self.item_matrix = nn.Embedding(num_items, latent_dim)

    def forward(self, user_ids, item_ids):
        # Calculate predicted scores by performing a dot product between the user and item embeddings
        user_embeds = self.user_matrix(user_ids)
        item_embeds = self.item_matrix(item_ids)

        # Dot product between user and item embeddings to get predicted score
        predicted_scores = (user_embeds * item_embeds).sum(dim=1)

        return predicted_scores


def initialize_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)




class MFAdvanced(nn.Module):
    """ Matrix factorization + user & item bias, weight init., sigmoid_range """
    def __init__(self, num_users, num_items, emb_dim, init, bias, sigmoid):
        super().__init__()
        self.bias = bias
        self.sigmoid = sigmoid
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        if bias:
            self.user_bias = nn.Parameter(torch.zeros(num_users))
            self.item_bias = nn.Parameter(torch.zeros(num_items))
            self.offset = nn.Parameter(torch.zeros(1))
        if init:
            self.user_emb.weight.data.uniform_(0., 0.05)
            self.item_emb.weight.data.uniform_(0., 0.05)
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        element_product = (user_emb*item_emb).sum(1)
        if self.bias:
            user_b = self.user_bias[user]
            item_b = self.item_bias[item]
            element_product += user_b + item_b + self.offset
        if self.sigmoid:
            return torch.sigmoid(element_product)
        return element_product