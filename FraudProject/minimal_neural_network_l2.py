"""Copy of the minimal neural network with L2 regularization."""

import torch
from torch import nn
from torch.nn import functional as F


def activation(x):
    return  torch.sigmoid(x)  #*x Custom activation function


def loss_fn(logits, target, pos_weight=100.0):
    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=logits.new_tensor([pos_weight]),
    )


class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        sizes = [input_size, *hidden_sizes, 1]
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = activation(layer(x))
        return self.layers[-1](x)


def train(
    X,
    y,
    hidden_sizes=(4,),
    epochs=25,
    learning_rate=0.01,
    pos_weight=100.0,
    seed=42,
    l2=1e-4,
):
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)
    model = Network(X.shape[1], hidden_sizes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2
    )

    for _ in range(epochs):
        loss = loss_fn(model(X), y, pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
