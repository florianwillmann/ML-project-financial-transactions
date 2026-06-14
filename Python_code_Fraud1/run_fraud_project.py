"""Select hyperparameters on validation data, then evaluate once on test data."""

from minimal_neural_network import train
from module1_data import load_data
from module3_evaluation import evaluate


DATA_FILE = "transactions.zip"
CONFIGS = [
    {"hidden_sizes": (), "epochs": 50, "learning_rate": 0.01, "pos_weight": 100},
    {"hidden_sizes": (16,), "epochs": 25, "learning_rate": 0.01, "pos_weight": 100},
    {"hidden_sizes": (32, 16), "epochs": 25, "learning_rate": 0.01, "pos_weight": 100},
    {"hidden_sizes": (64, 32), "epochs": 25, "learning_rate": 0.01, "pos_weight": 100},
    {"hidden_sizes": (96, 48), "epochs": 25, "learning_rate": 0.01, "pos_weight": 100},
    {"hidden_sizes": (32, 16), "epochs": 50, "learning_rate": 0.005, "pos_weight": 160},
    {"hidden_sizes": (64, 32), "epochs": 50, "learning_rate": 0.005, "pos_weight": 160},
    {"hidden_sizes": (64*2,64, 32), "epochs": 50, "learning_rate": 0.005, "pos_weight": 160}
]

data = load_data(DATA_FILE)
best_config = best_metrics = best_model = None

for config in CONFIGS:
    model = train(data["X_subtrain"], data["y_subtrain"], **config)
    _, metrics = evaluate(
        model, data["X_validation"], data["y_validation"], print_results=False
    )
    print(
        f"{config} -> validation ROC-AUC={metrics['roc_auc']:.4f}, "
        f"PR-AUC={metrics['pr_auc']:.4f}"
    )
    if best_metrics is None or metrics["roc_auc"] > best_metrics["roc_auc"]:
        best_config, best_metrics, best_model = config, metrics, model

print(f"\nSelected configuration: {best_config}")
_, subtrain_metrics = evaluate(
    best_model, data["X_subtrain"], data["y_subtrain"], print_results=False
)
print(
    f"Subtraining ROC-AUC={subtrain_metrics['roc_auc']:.4f}, "
    f"PR-AUC={subtrain_metrics['pr_auc']:.4f}"
)
print(
    f"Validation ROC-AUC={best_metrics['roc_auc']:.4f}, "
    f"PR-AUC={best_metrics['pr_auc']:.4f}"
)
print(
    f"Subtraining/validation gaps: ROC-AUC="
    f"{abs(subtrain_metrics['roc_auc'] - best_metrics['roc_auc']):.4f}, "
    f"PR-AUC={abs(subtrain_metrics['pr_auc'] - best_metrics['pr_auc']):.4f}"
)

final_model = train(data["X_train"], data["y_train"], **best_config)
_, test_metrics = evaluate(
    final_model, data["X_test"], data["y_test"], print_results=False
)

print(
    f"Test ROC-AUC={test_metrics['roc_auc']:.4f}, "
    f"PR-AUC={test_metrics['pr_auc']:.4f}"
)
print(
    f"Validation/test gaps: ROC-AUC="
    f"{abs(best_metrics['roc_auc'] - test_metrics['roc_auc']):.4f}, "
    f"PR-AUC={abs(best_metrics['pr_auc'] - test_metrics['pr_auc']):.4f}"
)
