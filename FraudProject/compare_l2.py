"""Compare the original network with different L2 strengths."""

import torch

from minimal_neural_network import train as train_original
from minimal_neural_network_l2 import train as train_l2
from module1_data import load_data
from module3_evaluation import evaluate


data = load_data("transactions.zip")
config = {
    "hidden_sizes": (64, 32),
    "epochs": 50,
    "learning_rate": 0.005,
    "pos_weight": 160,
}

models = [("original", train_original(data["X_subtrain"], data["y_subtrain"], **config))]
for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
    models.append(
        (
            f"L2={l2}",
            train_l2(data["X_subtrain"], data["y_subtrain"], **config, l2=l2),
        )
    )

results = []
for name, model in models:
    _, train_metrics = evaluate(
        model, data["X_subtrain"], data["y_subtrain"], print_results=False
    )
    _, validation_metrics = evaluate(
        model, data["X_validation"], data["y_validation"], print_results=False
    )
    results.append((name, validation_metrics))
    print(
        f"{name:10s} train ROC={train_metrics['roc_auc']:.4f}, "
        f"validation ROC={validation_metrics['roc_auc']:.4f}, "
        f"validation PR={validation_metrics['pr_auc']:.4f}"
    )

best_name, best_metrics = max(results, key=lambda result: result[1]["roc_auc"])
best_l2 = 0.0 if best_name == "original" else float(best_name.split("=")[1])
final_model = (
    train_original(data["X_train"], data["y_train"], **config)
    if best_l2 == 0.0
    else train_l2(data["X_train"], data["y_train"], **config, l2=best_l2)
)
_, test_metrics = evaluate(
    final_model, data["X_test"], data["y_test"], print_results=False
)

torch.save(
    {
        "model_state_dict": final_model.state_dict(),
        "input_size": data["X_train"].shape[1],
        "hidden_sizes": config["hidden_sizes"],
        "training_config": {**config, "l2": best_l2},
        "feature_columns": data["feature_columns"],
        "scaler_mean": data["scaler"].mean_,
        "scaler_scale": data["scaler"].scale_,
        "validation_metrics": best_metrics,
        "test_metrics": test_metrics,
    },
    "best_validation_model.pt",
)

print(f"\nSelected: {best_name}")
print(f"Validation ROC-AUC: {best_metrics['roc_auc']:.4f}")
print(f"Test ROC-AUC:       {test_metrics['roc_auc']:.4f}")
print(f"Test PR-AUC:        {test_metrics['pr_auc']:.4f}")
print("Saved: best_validation_model.pt")
