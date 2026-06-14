"""Compare every hyperparameter configuration on validation and test data."""

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
    {
        "hidden_sizes": (64 * 2, 64, 32),
        "epochs": 50,
        "learning_rate": 0.005,
        "pos_weight": 160,
    },
]


def format_score(value):
    return f"{value:.4f}"


data = load_data(DATA_FILE)
results = []

print(
    "Auswahl | Val ROC  Test ROC  Abstand | "
    "Val PR   Test PR   Abstand | Val F1   Test F1"
)
print("-" * 88)

for selection, config in enumerate(CONFIGS, start=1):
    validation_model = train(
        data["X_subtrain"],
        data["y_subtrain"],
        **config,
    )
    _, validation_metrics = evaluate(
        validation_model,
        data["X_validation"],
        data["y_validation"],
        print_results=False,
    )

    # The test data uses the scaler fitted on the complete training set.
    test_model = train(
        data["X_train"],
        data["y_train"],
        **config,
    )
    _, test_metrics = evaluate(
        test_model,
        data["X_test"],
        data["y_test"],
        print_results=False,
    )

    roc_gap = abs(validation_metrics["roc_auc"] - test_metrics["roc_auc"])
    pr_gap = abs(validation_metrics["pr_auc"] - test_metrics["pr_auc"])
    results.append(
        {
            "selection": selection,
            "config": config,
            "validation": validation_metrics,
            "test": test_metrics,
        }
    )

    print(
        f"{selection:>7} | "
        f"{format_score(validation_metrics['roc_auc']):>7}  "
        f"{format_score(test_metrics['roc_auc']):>8}  "
        f"{format_score(roc_gap):>7} | "
        f"{format_score(validation_metrics['pr_auc']):>7}  "
        f"{format_score(test_metrics['pr_auc']):>8}  "
        f"{format_score(pr_gap):>7} | "
        f"{format_score(validation_metrics['f1']):>7}  "
        f"{format_score(test_metrics['f1']):>7}"
    )

best_result = max(results, key=lambda result: result["validation"]["roc_auc"])

print(
    "\nBeste Auswahl anhand der Validierungs-ROC-AUC: "
    f"{best_result['selection']}"
)
print(
    "Validierung: "
    f"ROC-AUC={best_result['validation']['roc_auc']:.4f}, "
    f"PR-AUC={best_result['validation']['pr_auc']:.4f}, "
    f"F1={best_result['validation']['f1']:.4f}"
)
print(
    "Test:         "
    f"ROC-AUC={best_result['test']['roc_auc']:.4f}, "
    f"PR-AUC={best_result['test']['pr_auc']:.4f}, "
    f"F1={best_result['test']['f1']:.4f}"
)
