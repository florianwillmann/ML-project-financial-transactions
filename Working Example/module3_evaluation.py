"""Module 3: Evaluate the neural network on unseen test data."""

import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(model, X_test, y_test, threshold=0.5, print_results=True):
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_test))
        scores = torch.sigmoid(logits).flatten().numpy()

    predictions = (scores >= threshold).astype(int)
    metrics = {
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    if print_results:
        for name, value in metrics.items():
            print(f"{name}: {value}")
    return scores, metrics
