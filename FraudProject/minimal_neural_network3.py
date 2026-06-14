"""Configurable neural network used by the Optuna search."""

import torch
from torch import nn
from torch.nn import functional as F


ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
}


class Network(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation="relu", dropout=0.0):
        super().__init__()
        layers = []
        sizes = [input_size, *hidden_sizes]

        for input_features, output_features in zip(sizes, sizes[1:]):
            layers.extend(
                [
                    nn.Linear(input_features, output_features),
                    ACTIVATIONS[activation](),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(sizes[-1], 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def train(
    X,
    y,
    hidden_sizes=(32,),
    activation="relu",
    dropout=0.0,
    epochs=40,
    learning_rate=0.001,
    pos_weight=100.0,
    weight_decay=0.0,
    seed=42,
):
    torch.manual_seed(seed)
    X = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1)

    model = Network(X.shape[1], hidden_sizes, activation, dropout)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    model.train()
    for _ in range(epochs):
        logits = model(X)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            y,
            pos_weight=logits.new_tensor([pos_weight]),
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
