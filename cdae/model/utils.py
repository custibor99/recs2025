import torch
import torch.nn as nn


def _get_activation(name: str):
    if name.lower() == "sigmoid":
        return torch.sigmoid
    elif name.lower() == "tanh":
        return torch.tanh
    elif name.lower() == "relu6":
        return nn.ReLU6()
    elif name.lower() == "relu":
        return torch.relu
    elif name.lower() == "elu":
        return torch.elu
    else:
        raise ValueError(f"Unsupported activation: {name}.")


def _get_optimizer(name: str):
    if name.lower() == "adam":
        return torch.optim.Adam
    elif name.lower() == "adagrad":
        return torch.optim.Adagrad
    elif name.lower() == "rmsprop":
        return torch.optim.RMSprop
    elif name.lower() == "sgd":
        return torch.optim.SGD
    else:
        raise ValueError(f"Unsupported optimizer: {name}.")
