"""Professional but compact hyperparameter search with Optuna."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import precision_recall_curve

from minimal_neural_network3 import train
from module1_data import load_data
from module3_evaluation import evaluate


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "transactions.zip"
OUTPUT_DIR = PROJECT_DIR / "optuna_results"
N_TRIALS = 25
SEED = 42

ARCHITECTURES = {
    "linear": (),
    "small": (16,),
    "medium": (32,),
    "large": (64,),
    "two_layers": (64, 32),
    "three_layers": (128, 64, 32),
}


def config_from_params(params):
    return {
        "hidden_sizes": ARCHITECTURES[params["architecture"]],
        "activation": params["activation"],
        "dropout": params["dropout"],
        "epochs": params["epochs"],
        "learning_rate": params["learning_rate"],
        "pos_weight": params["pos_weight"],
        "weight_decay": params["weight_decay"],
        "seed": SEED,
    }


def best_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1_scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    return float(thresholds[np.argmax(f1_scores)])


def create_objective(data):
    def objective(trial):
        params = {
            "architecture": trial.suggest_categorical(
                "architecture", list(ARCHITECTURES)
            ),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "tanh", "sigmoid"]
            ),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4, step=0.1),
            "epochs": trial.suggest_int("epochs", 20, 80, step=10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-4, 2e-2, log=True
            ),
            "pos_weight": trial.suggest_float("pos_weight", 20, 250, log=True),
            "weight_decay": trial.suggest_float(
                "weight_decay", 1e-7, 1e-2, log=True
            ),
        }

        model = train(data["X_subtrain"], data["y_subtrain"], **config_from_params(params))
        scores, _ = evaluate(
            model,
            data["X_validation"],
            data["y_validation"],
            print_results=False,
        )
        threshold = best_f1_threshold(data["y_validation"], scores)
        _, metrics = evaluate(
            model,
            data["X_validation"],
            data["y_validation"],
            threshold=threshold,
            print_results=False,
        )

        trial.set_user_attr("roc_auc", metrics["roc_auc"])
        trial.set_user_attr("f1", metrics["f1"])
        trial.set_user_attr("threshold", threshold)
        return metrics["pr_auc"]

    return objective


def trial_table(study):
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append(
            {
                "trial": trial.number,
                "validation_pr_auc": trial.value,
                "validation_roc_auc": trial.user_attrs["roc_auc"],
                "validation_f1": trial.user_attrs["f1"],
                "threshold": trial.user_attrs["threshold"],
                **trial.params,
            }
        )
    return pd.DataFrame(rows).sort_values("validation_pr_auc", ascending=False)


def save_plots(results, validation_metrics, test_metrics):
    ordered = results.sort_values("trial")
    best_so_far = ordered["validation_pr_auc"].cummax()

    plt.figure(figsize=(9, 5))
    plt.plot(ordered["trial"], ordered["validation_pr_auc"], "o", label="Trial")
    plt.plot(ordered["trial"], best_so_far, label="Bestwert bisher")
    plt.xlabel("Optuna-Trial")
    plt.ylabel("Validierungs-PR-AUC")
    plt.title("Verlauf der Hyperparameteroptimierung")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "optimization_history.png", dpi=160)
    plt.close()

    metric_names = ["pr_auc", "roc_auc", "f1"]
    labels = ["PR-AUC", "ROC-AUC", "F1"]
    x = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    plt.bar(
        x - 0.18,
        [validation_metrics[name] for name in metric_names],
        width=0.36,
        label="Validierung",
    )
    plt.bar(
        x + 0.18,
        [test_metrics[name] for name in metric_names],
        width=0.36,
        label="Test",
    )
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Gewaehltes Modell: Validierung und Test")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_vs_test.png", dpi=160)
    plt.close()


def json_metrics(metrics):
    return {
        name: value if name == "confusion_matrix" else float(value)
        for name, value in metrics.items()
    }


def print_summary(results, best_config, validation_metrics, test_metrics, data):
    columns = [
        "trial",
        "validation_pr_auc",
        "validation_roc_auc",
        "validation_f1",
        "architecture",
        "activation",
        "epochs",
        "learning_rate",
        "pos_weight",
    ]
    print("\nBeste 10 Versuche (sortiert nach Validierungs-PR-AUC)")
    print("-" * 110)
    print(
        results[columns].head(10).to_string(
            index=False,
            formatters={
                "validation_pr_auc": "{:.4f}".format,
                "validation_roc_auc": "{:.4f}".format,
                "validation_f1": "{:.4f}".format,
                "learning_rate": "{:.5f}".format,
                "pos_weight": "{:.1f}".format,
            },
        )
    )

    validation_rate = float(np.mean(data["y_validation"]))
    test_rate = float(np.mean(data["y_test"]))

    print("\nAusgewaehlte Konfiguration")
    print("-" * 35)
    for name, value in best_config.items():
        if name != "seed":
            print(f"{name:>15}: {value}")

    print("\nAbschliessende Bewertung")
    print("-" * 70)
    print("Datensatz       PR-AUC   ROC-AUC      F1   Fraud-Rate   PR-Lift")
    print(
        f"Validierung     {validation_metrics['pr_auc']:.4f}    "
        f"{validation_metrics['roc_auc']:.4f}  "
        f"{validation_metrics['f1']:.4f}      "
        f"{validation_rate:.4f}      "
        f"{validation_metrics['pr_auc'] / validation_rate:.1f}x"
    )
    print(
        f"Test            {test_metrics['pr_auc']:.4f}    "
        f"{test_metrics['roc_auc']:.4f}  "
        f"{test_metrics['f1']:.4f}      "
        f"{test_rate:.4f}      "
        f"{test_metrics['pr_auc'] / test_rate:.1f}x"
    )
    print(
        "Absoluter Gap   "
        f"{abs(validation_metrics['pr_auc'] - test_metrics['pr_auc']):.4f}    "
        f"{abs(validation_metrics['roc_auc'] - test_metrics['roc_auc']):.4f}  "
        f"{abs(validation_metrics['f1'] - test_metrics['f1']):.4f}"
    )
    print(f"Test-Konfusionsmatrix: {test_metrics['confusion_matrix']}")
    print(f"\nDetailergebnisse: {OUTPUT_DIR}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = load_data(DATA_FILE)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="fraud_pr_auc_search",
    )

    print(
        f"Optuna optimiert {N_TRIALS} Konfigurationen auf PR-AUC. "
        "Das Testset bleibt bis zum Schluss unangetastet."
    )
    study.optimize(create_objective(data), n_trials=N_TRIALS, show_progress_bar=True)

    results = trial_table(study)
    results.to_csv(OUTPUT_DIR / "all_trials.csv", index=False)

    best_trial = study.best_trial
    best_config = config_from_params(best_trial.params)
    threshold = best_trial.user_attrs["threshold"]

    validation_model = train(
        data["X_subtrain"],
        data["y_subtrain"],
        **best_config,
    )
    _, validation_metrics = evaluate(
        validation_model,
        data["X_validation"],
        data["y_validation"],
        threshold=threshold,
        print_results=False,
    )

    final_model = train(data["X_train"], data["y_train"], **best_config)
    _, test_metrics = evaluate(
        final_model,
        data["X_test"],
        data["y_test"],
        threshold=threshold,
        print_results=False,
    )

    summary = {
        "best_trial": best_trial.number,
        "best_config": {
            **best_config,
            "hidden_sizes": list(best_config["hidden_sizes"]),
        },
        "threshold": threshold,
        "validation_metrics": json_metrics(validation_metrics),
        "test_metrics": json_metrics(test_metrics),
    }
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    save_plots(results, validation_metrics, test_metrics)
    print_summary(results, best_config, validation_metrics, test_metrics, data)


if __name__ == "__main__":
    main()
