"""Train the NumPy neural network used for fraud scoring.

Open this file in Spyder and run it to retrain the model from transactions.csv.zip.
The script optimizes weighted binary cross-entropy and selects the model with the
best validation ROC-AUC, because the leaderboard score is roc_auc_score.

Outputs:
    trained_nn_model.json  - learned weights and preprocessing statistics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def manual_roc_auc_score(y_true, scores):
    """Fallback ROC-AUC implementation for environments without scikit-learn."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)

    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def auc_score(y_true, scores):
    if roc_auc_score is not None:
        return float(roc_auc_score(y_true, scores))
    return float(manual_roc_auc_score(y_true, scores))


def make_features(data):
    """Apply the same feature transformations used during prediction."""
    X = data.drop(columns=["Class"]).copy() if "Class" in data.columns else data.copy()
    X["Amount"] = np.log1p(X["Amount"].astype(float))
    X["Time"] = X["Time"].astype(float) / 172800.0
    return X.astype(float)


def stratified_train_val_split(y, validation_fraction=0.2, seed=123):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    rng.shuffle(positive)
    rng.shuffle(negative)

    n_pos_val = max(1, int(validation_fraction * len(positive)))
    n_neg_val = int(validation_fraction * len(negative))

    val_idx = np.r_[positive[:n_pos_val], negative[:n_neg_val]]
    train_idx = np.r_[positive[n_pos_val:], negative[n_neg_val:]]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def forward(X, W1, b1, W2, b2, W3, b3):
    z1 = X @ W1 + b1
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ W2 + b2
    a2 = np.maximum(z2, 0.0)
    logits = a2 @ W3 + b3
    return sigmoid(logits).ravel()


def train_neural_network(
    X_train,
    y_train,
    X_val,
    y_val,
    seed=44,
    pos_weight=350.0,
    hidden1=64,
    hidden2=32,
    epochs=55,
    batch_size=2048,
    learning_rate=7e-4,
    l2=2e-5,
    patience=12,
):
    """Train a two-hidden-layer ReLU neural network with Adam."""
    rng = np.random.default_rng(seed)
    n_features = X_train.shape[1]

    W1 = rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden1)).astype(np.float32)
    b1 = np.zeros(hidden1, dtype=np.float32)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden1), (hidden1, hidden2)).astype(np.float32)
    b2 = np.zeros(hidden2, dtype=np.float32)
    W3 = rng.normal(0, np.sqrt(1.0 / hidden2), (hidden2, 1)).astype(np.float32)
    b3 = np.zeros(1, dtype=np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    step = 0

    best_auc = -1.0
    best_params = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_train))

        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            xb = X_train[idx]
            yb = y_train[idx].reshape(-1, 1)

            z1 = xb @ W1 + b1
            a1 = np.maximum(z1, 0.0)
            z2 = a1 @ W2 + b2
            a2 = np.maximum(z2, 0.0)
            probabilities = sigmoid(a2 @ W3 + b3)

            sample_weights = 1.0 + (pos_weight - 1.0) * yb
            dz3 = (probabilities - yb) * sample_weights / len(idx)
            dW3 = a2.T @ dz3 + l2 * W3
            db3 = dz3.sum(axis=0)

            dz2 = (dz3 @ W3.T) * (z2 > 0)
            dW2 = a1.T @ dz2 + l2 * W2
            db2 = dz2.sum(axis=0)

            dz1 = (dz2 @ W2.T) * (z1 > 0)
            dW1 = xb.T @ dz1 + l2 * W1
            db1 = dz1.sum(axis=0)

            gradients = [dW1, db1, dW2, db2, dW3, db3]
            step += 1
            for i, (param, grad) in enumerate(zip(params, gradients)):
                m[i] = beta1 * m[i] + (1.0 - beta1) * grad
                v[i] = beta2 * v[i] + (1.0 - beta2) * (grad * grad)
                m_hat = m[i] / (1.0 - beta1 ** step)
                v_hat = v[i] / (1.0 - beta2 ** step)
                param -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)

        val_scores = forward(X_val, W1, b1, W2, b2, W3, b3)
        val_auc = auc_score(y_val, val_scores)
        print(f"Epoch {epoch:03d} validation ROC-AUC: {val_auc:.6f}")

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_params = [p.copy() for p in params]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    return best_auc, best_params


def save_model(output_path, columns, mean, std, params, validation_auc, train_auc, args):
    W1, b1, W2, b2, W3, b3 = params
    model = {
        "columns": columns,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "W1": W1.tolist(),
        "b1": b1.tolist(),
        "W2": W2.tolist(),
        "b2": b2.tolist(),
        "W3": W3.ravel().tolist(),
        "b3": b3.tolist(),
        "validation_auc": float(validation_auc),
        "train_auc": float(train_auc),
        "training_config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "feature_notes": {
            "Amount": "log1p before standardization",
            "Time": "divided by 172800 before standardization",
        },
    }
    output_path.write_text(json.dumps(model), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a neural network for fraud ROC-AUC scoring.")
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("transactions.csv.zip"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("trained_nn_model.json"))
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--pos-weight", type=float, default=350.0)
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--hidden1", type=int, default=64)
    parser.add_argument("--hidden2", type=int, default=32)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.data}")
    data = pd.read_csv(args.data)
    X_frame = make_features(data)
    y = data["Class"].to_numpy(dtype=np.float32)

    train_idx, val_idx = stratified_train_val_split(y, args.validation_fraction, args.split_seed)
    raw_X = X_frame.to_numpy(dtype=float)
    mean = raw_X[train_idx].mean(axis=0)
    std = raw_X[train_idx].std(axis=0)
    std[std == 0] = 1.0
    X = ((raw_X - mean) / std).astype(np.float32)

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    print(f"Train rows: {len(train_idx)}")
    print(f"Validation rows: {len(val_idx)}")
    print(f"Fraud rate: {y.mean():.6f}")

    validation_auc, best_params = train_neural_network(
        X_train,
        y_train,
        X_val,
        y_val,
        seed=args.seed,
        pos_weight=args.pos_weight,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    train_scores = forward(X, *best_params)
    train_auc = auc_score(y, train_scores)
    print(f"Best validation ROC-AUC: {validation_auc:.6f}")
    print(f"Full training-data ROC-AUC: {train_auc:.6f}")

    save_model(
        args.output,
        list(X_frame.columns),
        mean,
        std,
        best_params,
        validation_auc,
        train_auc,
        args,
    )
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
