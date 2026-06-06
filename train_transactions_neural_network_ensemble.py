from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_COLUMN = "Class"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models/transactions_nn_ensemble.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "artifacts/transactions_nn_ensemble_metrics.json"
PARAMETER_NAMES = ("W1", "b1", "W2", "b2", "W3", "b3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a stronger neural-network fraud detector with a small, bounded "
            "validation-only hyperparameter search."
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional path to transactions.csv.zip or transactions.zip.",
    )
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.2,
        help="Fraction of the labeled data reserved for validation.",
    )
    parser.add_argument(
        "--search-level",
        choices=["minimal", "standard", "extended"],
        default="standard",
        help="How many hand-picked neural-network configurations to try.",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=3,
        help="How many random seeds to average after choosing the best configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE,
        help="Base random seed for initialization and splits.",
    )
    parser.add_argument(
        "--skip-full-retrain",
        action="store_true",
        help=(
            "Keep the final artifact on the train/validation split instead of "
            "retraining the chosen ensemble on the full labeled dataset."
        ),
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to store the metrics JSON.",
    )
    return parser.parse_args()


def resolve_data_path(explicit_path: str | None) -> Path:
    candidates: list[Path] = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    candidates.extend(
        [
            Path("/data/mlproject22/transactions.csv.zip"),
            BASE_DIR / "transactions.csv.zip",
            BASE_DIR / "transactions.csv (1).zip",
            BASE_DIR / "transactions.zip",
            Path("transactions.csv.zip"),
            Path("transactions.csv (1).zip"),
            Path("transactions.zip"),
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find the transactions dataset. Looked in:\n{searched}")


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(data_path)
    X = frame.drop(columns=TARGET_COLUMN)
    y = frame[TARGET_COLUMN].astype(int)
    return X, y


def build_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.copy()

    if TARGET_COLUMN in values.columns:
        values = values.drop(columns=TARGET_COLUMN)

    time_seconds = values["Time"].astype(float)
    amount = values["Amount"].astype(float)

    values["TimeDays"] = time_seconds / 86400.0#86400.0 are the seconds of a day
    values["TimeSinDay"] = np.sin((2.0 * np.pi * time_seconds) / 86400.0)# sin and cos are used because they are periodic (like time) and
    values["TimeCosDay"] = np.cos((2.0 * np.pi * time_seconds) / 86400.0)# differientiable, with very regular propertys
    values["LogAmount"] = np.log1p(amount)
    values["AmountIsZero"] = (amount == 0.0).astype(float)

    return values.astype(float)


def fit_standardization(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = train_frame.mean(axis=0).to_numpy(dtype=np.float32)
    std = train_frame.std(axis=0).to_numpy(dtype=np.float32)
    std[std == 0.0] = 1.0
    return mean, std


def transform_features(frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    values = frame.to_numpy(dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -50.0, 50.0)))


def leaky_relu(values: np.ndarray, negative_slope: float) -> np.ndarray:
    return np.where(values > 0.0, values, negative_slope * values)


def leaky_relu_grad(values: np.ndarray, negative_slope: float) -> np.ndarray:
    return np.where(values > 0.0, 1.0, negative_slope).astype(np.float32)


def initialize_parameters(
    input_dim: int,
    hidden1: int,
    hidden2: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        "W1": rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden1)).astype(np.float32),
        "b1": np.zeros(hidden1, dtype=np.float32),
        "W2": rng.normal(0.0, np.sqrt(2.0 / hidden1), (hidden1, hidden2)).astype(np.float32),
        "b2": np.zeros(hidden2, dtype=np.float32),
        "W3": rng.normal(0.0, np.sqrt(1.0 / hidden2), (hidden2, 1)).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }


def copy_parameters(parameters: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: value.copy() for name, value in parameters.items()}


def forward_pass(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
    negative_slope: float,
    dropout_rate: float,
    rng: np.random.Generator | None = None,
    training: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z1 = X @ parameters["W1"] + parameters["b1"]
    a1 = leaky_relu(z1, negative_slope).astype(np.float32)
    mask1 = None

    if training and dropout_rate > 0.0:
        if rng is None:
            raise ValueError("A random generator is required when dropout is enabled.")
        mask1 = (rng.random(a1.shape, dtype=np.float32) >= dropout_rate).astype(np.float32)
        mask1 /= 1.0 - dropout_rate
        a1 *= mask1

    z2 = a1 @ parameters["W2"] + parameters["b2"]
    a2 = leaky_relu(z2, negative_slope).astype(np.float32)
    mask2 = None

    if training and dropout_rate > 0.0:
        if rng is None:
            raise ValueError("A random generator is required when dropout is enabled.")
        mask2 = (rng.random(a2.shape, dtype=np.float32) >= dropout_rate).astype(np.float32)
        mask2 /= 1.0 - dropout_rate
        a2 *= mask2

    logits = a2 @ parameters["W3"] + parameters["b3"]
    probabilities = sigmoid(logits).astype(np.float32)

    cache = {
        "X": X,
        "z1": z1,
        "a1": a1,
        "mask1": mask1,
        "z2": z2,
        "a2": a2,
        "mask2": mask2,
        "probabilities": probabilities,
    }
    return probabilities.ravel(), cache


def compute_gradients(
    y_batch: np.ndarray,
    cache: dict[str, np.ndarray],
    parameters: dict[str, np.ndarray],
    config: dict[str, float | int | str],
) -> dict[str, np.ndarray]:
    negative_slope = float(config["negative_slope"])
    pos_weight = float(config["pos_weight"])
    l2 = float(config["l2"])

    y_column = y_batch.reshape(-1, 1).astype(np.float32)
    probabilities = cache["probabilities"]

    sample_weights = np.where(y_column > 0.5, pos_weight, 1.0).astype(np.float32)
    sample_weights /= sample_weights.mean()

    dz3 = (probabilities - y_column) * sample_weights / y_column.shape[0]
    dW3 = cache["a2"].T @ dz3 + l2 * parameters["W3"]
    db3 = dz3.sum(axis=0)

    da2 = dz3 @ parameters["W3"].T
    if cache["mask2"] is not None:
        da2 *= cache["mask2"]
    dz2 = da2 * leaky_relu_grad(cache["z2"], negative_slope)
    dW2 = cache["a1"].T @ dz2 + l2 * parameters["W2"]
    db2 = dz2.sum(axis=0)

    da1 = dz2 @ parameters["W2"].T
    if cache["mask1"] is not None:
        da1 *= cache["mask1"]
    dz1 = da1 * leaky_relu_grad(cache["z1"], negative_slope)
    dW1 = cache["X"].T @ dz1 + l2 * parameters["W1"]
    db1 = dz1.sum(axis=0)

    gradients = {
        "W1": dW1.astype(np.float32),
        "b1": db1.astype(np.float32),
        "W2": dW2.astype(np.float32),
        "b2": db2.astype(np.float32),
        "W3": dW3.astype(np.float32),
        "b3": db3.astype(np.float32),
    }

    clip_gradients(gradients, max_norm=float(config["grad_clip"]))
    return gradients


def clip_gradients(gradients: dict[str, np.ndarray], max_norm: float) -> None:
    squared_norm = 0.0
    for gradient in gradients.values():
        squared_norm += float(np.sum(gradient * gradient))

    total_norm = np.sqrt(squared_norm)
    if total_norm <= max_norm or total_norm == 0.0:
        return

    scale = max_norm / (total_norm + 1e-12)
    for name in gradients:
        gradients[name] *= scale


def initialize_optimizer(parameters: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray] | int]:
    return {
        "m": {name: np.zeros_like(value) for name, value in parameters.items()},
        "v": {name: np.zeros_like(value) for name, value in parameters.items()},
        "step": 0,
    }


def adam_update(
    parameters: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    optimizer: dict[str, dict[str, np.ndarray] | int],
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> None:
    optimizer["step"] = int(optimizer["step"]) + 1
    step = int(optimizer["step"])
    m_state = optimizer["m"]
    v_state = optimizer["v"]

    for name in PARAMETER_NAMES:
        m_state[name] = beta1 * m_state[name] + (1.0 - beta1) * gradients[name]
        v_state[name] = beta2 * v_state[name] + (1.0 - beta2) * (gradients[name] * gradients[name])

        m_hat = m_state[name] / (1.0 - beta1**step)
        v_hat = v_state[name] / (1.0 - beta2**step)
        parameters[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if thresholds.size == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    f1_values = 2.0 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_values))

    return float(thresholds[best_index]), {
        "f1": float(f1_values[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
    }


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | list[list[int]]]:
    predicted_labels = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, predicted_labels, labels=[0, 1])

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, predicted_labels, zero_division=0)),
        "recall": float(recall_score(y_true, predicted_labels, zero_division=0)),
        "f1": float(f1_score(y_true, predicted_labels, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": cm.astype(int).tolist(),
    }


def train_network(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, float | int | str],
    seed: int,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    fixed_epochs: int | None = None,
    verbose: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    parameters = initialize_parameters(
        input_dim=X_train.shape[1],
        hidden1=int(config["hidden1"]),
        hidden2=int(config["hidden2"]),
        rng=rng,
    )
    optimizer = initialize_optimizer(parameters)

    max_epochs = fixed_epochs if fixed_epochs is not None else int(config["epochs"])
    batch_size = int(config["batch_size"])
    dropout_rate = float(config["dropout"])
    negative_slope = float(config["negative_slope"])
    patience = int(config["patience"])
    lr_patience = int(config["lr_patience"])
    lr_decay = float(config["lr_decay"])
    min_learning_rate = float(config["min_learning_rate"])
    current_learning_rate = float(config["learning_rate"])

    history: list[dict[str, float | int]] = []
    best_parameters: dict[str, np.ndarray] | None = None
    best_epoch = 0
    best_auc = -np.inf
    best_pr_auc = -np.inf
    epochs_without_improvement = 0
    epochs_since_lr_drop = 0
    monitor_size = min(20000, X_train.shape[0])
    monitor_index = rng.choice(X_train.shape[0], size=monitor_size, replace=False)
    X_monitor = X_train[monitor_index]
    y_monitor = y_train[monitor_index]

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(X_train.shape[0])

        for start in range(0, order.shape[0], batch_size):
            batch_index = order[start:start + batch_size]
            X_batch = X_train[batch_index]
            y_batch = y_train[batch_index]
            _, cache = forward_pass(
                X_batch,
                parameters,
                negative_slope=negative_slope,
                dropout_rate=dropout_rate,
                rng=rng,
                training=True,
            )
            gradients = compute_gradients(y_batch, cache, parameters, config)
            adam_update(parameters, gradients, optimizer, learning_rate=current_learning_rate)

        monitor_scores = predict_with_parameters(X_monitor, parameters, negative_slope)
        train_auc = float(roc_auc_score(y_monitor, monitor_scores))

        history_entry: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": float(current_learning_rate),
            "train_roc_auc": train_auc,
        }

        if X_val is not None and y_val is not None:
            val_scores = predict_with_parameters(X_val, parameters, negative_slope)
            val_auc = float(roc_auc_score(y_val, val_scores))
            val_pr_auc = float(average_precision_score(y_val, val_scores))
            history_entry["validation_roc_auc"] = val_auc
            history_entry["validation_pr_auc"] = val_pr_auc

            improved = (val_auc > best_auc + 1e-5) or (
                abs(val_auc - best_auc) <= 1e-5 and val_pr_auc > best_pr_auc + 1e-5
            )

            if verbose:
                print(
                    f"Epoch {epoch:03d} | train ROC-AUC {train_auc:.6f} | "
                    f"val ROC-AUC {val_auc:.6f} | val PR-AUC {val_pr_auc:.6f}"
                )

            if improved:
                best_auc = val_auc
                best_pr_auc = val_pr_auc
                best_epoch = epoch
                best_parameters = copy_parameters(parameters)
                epochs_without_improvement = 0
                epochs_since_lr_drop = 0
            else:
                epochs_without_improvement += 1
                epochs_since_lr_drop += 1

                if epochs_since_lr_drop >= lr_patience and current_learning_rate > min_learning_rate:
                    current_learning_rate = max(current_learning_rate * lr_decay, min_learning_rate)
                    epochs_since_lr_drop = 0
                    if verbose:
                        print(f"Reducing learning rate to {current_learning_rate:.6g}")

                if fixed_epochs is None and epochs_without_improvement >= patience:
                    if verbose:
                        print(f"Early stopping after epoch {epoch}.")
                    history.append(history_entry)
                    break
        else:
            if verbose and (epoch == 1 or epoch == max_epochs or epoch % 10 == 0):
                print(f"Epoch {epoch:03d} | train ROC-AUC {train_auc:.6f}")

        history.append(history_entry)

    if best_parameters is None:
        best_parameters = copy_parameters(parameters)
        best_epoch = max_epochs

    result = {
        "parameters": best_parameters,
        "best_epoch": int(best_epoch),
        "history": history,
    }

    if X_val is not None and y_val is not None:
        train_scores = predict_with_parameters(X_train, best_parameters, negative_slope)
        val_scores = predict_with_parameters(X_val, best_parameters, negative_slope)
        result["train_scores"] = train_scores
        result["validation_scores"] = val_scores
        result["best_validation_roc_auc"] = float(roc_auc_score(y_val, val_scores))
        result["best_validation_pr_auc"] = float(average_precision_score(y_val, val_scores))

    return result


def predict_with_parameters(
    X: np.ndarray,
    parameters: dict[str, np.ndarray],
    negative_slope: float,
) -> np.ndarray:
    scores, _ = forward_pass(
        X,
        parameters,
        negative_slope=negative_slope,
        dropout_rate=0.0,
        rng=None,
        training=False,
    )
    return scores.astype(np.float64)


def average_model_scores(
    X: np.ndarray,
    models: list[dict[str, object]],
    negative_slope: float,
) -> np.ndarray:
    score_stack = [
        predict_with_parameters(X, model["parameters"], negative_slope)
        for model in models
    ]
    return np.mean(np.vstack(score_stack), axis=0)


def build_candidate_configs(search_level: str) -> list[dict[str, float | int | str]]:
    base_configs = [
        {
            "name": "compact_ranker",
            "hidden1": 64,
            "hidden2": 32,
            "dropout": 0.05,
            "learning_rate": 1.0e-3,
            "min_learning_rate": 1.5e-4,
            "lr_decay": 0.5,
            "lr_patience": 4,
            "l2": 2.0e-5,
            "pos_weight": 100.0,
            "batch_size": 2048,
            "epochs": 40,
            "patience": 10,
            "negative_slope": 0.05,
            "grad_clip": 5.0,
        },
        {
            "name": "balanced_ranker",
            "hidden1": 96,
            "hidden2": 48,
            "dropout": 0.10,
            "learning_rate": 8.0e-4,
            "min_learning_rate": 1.0e-4,
            "lr_decay": 0.5,
            "lr_patience": 4,
            "l2": 4.0e-5,
            "pos_weight": 160.0,
            "batch_size": 2048,
            "epochs": 48,
            "patience": 12,
            "negative_slope": 0.05,
            "grad_clip": 5.0,
        },
        {
            "name": "wide_ranker",
            "hidden1": 128,
            "hidden2": 64,
            "dropout": 0.15,
            "learning_rate": 7.0e-4,
            "min_learning_rate": 8.0e-5,
            "lr_decay": 0.5,
            "lr_patience": 5,
            "l2": 7.0e-5,
            "pos_weight": 220.0,
            "batch_size": 2048,
            "epochs": 54,
            "patience": 14,
            "negative_slope": 0.05,
            "grad_clip": 5.0,
        },
        {
            "name": "wide_low_weight",
            "hidden1": 128,
            "hidden2": 64,
            "dropout": 0.10,
            "learning_rate": 9.0e-4,
            "min_learning_rate": 1.2e-4,
            "lr_decay": 0.5,
            "lr_patience": 4,
            "l2": 5.0e-5,
            "pos_weight": 120.0,
            "batch_size": 2048,
            "epochs": 48,
            "patience": 12,
            "negative_slope": 0.05,
            "grad_clip": 5.0,
        },
    ]

    if search_level == "minimal":
        return base_configs[:2]

    if search_level == "extended":
        return base_configs + [
            {
                "name": "compact_heavier_weight",
                "hidden1": 64,
                "hidden2": 32,
                "dropout": 0.10,
                "learning_rate": 8.0e-4,
                "min_learning_rate": 1.0e-4,
                "lr_decay": 0.5,
                "lr_patience": 4,
                "l2": 6.0e-5,
                "pos_weight": 280.0,
                "batch_size": 2048,
                "epochs": 52,
                "patience": 14,
                "negative_slope": 0.05,
                "grad_clip": 5.0,
            },
            {
                "name": "wider_dropout",
                "hidden1": 160,
                "hidden2": 80,
                "dropout": 0.20,
                "learning_rate": 6.0e-4,
                "min_learning_rate": 8.0e-5,
                "lr_decay": 0.5,
                "lr_patience": 5,
                "l2": 8.0e-5,
                "pos_weight": 180.0,
                "batch_size": 2048,
                "epochs": 56,
                "patience": 14,
                "negative_slope": 0.05,
                "grad_clip": 5.0,
            },
        ]

    return base_configs


def summarize_trial(
    config: dict[str, float | int | str],
    training_result: dict[str, object],
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, object]:
    train_scores = np.asarray(training_result["train_scores"], dtype=float)
    val_scores = np.asarray(training_result["validation_scores"], dtype=float)
    best_threshold, threshold_summary = find_best_threshold(y_val, val_scores)

    return {
        "config": {key: value for key, value in config.items()},
        "best_epoch": int(training_result["best_epoch"]),
        "train_metrics_default_threshold": evaluate_scores(y_train, train_scores, threshold=0.5),
        "validation_metrics_default_threshold": evaluate_scores(y_val, val_scores, threshold=0.5),
        "validation_metrics_best_threshold": evaluate_scores(y_val, val_scores, threshold=best_threshold),
        "threshold_search": threshold_summary,
    }


def choose_best_trial(trials: list[dict[str, object]]) -> dict[str, object]:
    return max(
        trials,
        key=lambda trial: (
            float(trial["validation_metrics_default_threshold"]["roc_auc"]),
            float(trial["validation_metrics_default_threshold"]["pr_auc"]),
        ),
    )


def build_ensemble_seeds(base_seed: int, ensemble_size: int) -> list[int]:
    return [base_seed + 97 * offset for offset in range(ensemble_size)]


def to_serializable(value: object) -> object:
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_prediction_artifact(
    feature_columns: list[str],
    mean: np.ndarray,
    std: np.ndarray,
    models: list[dict[str, object]],
    config: dict[str, float | int | str],
    metrics: dict[str, object],
) -> dict[str, object]:
    compact_models = []

    for model in models:
        compact_models.append(
            {
                "seed": int(model["seed"]),
                "best_epoch": int(model["best_epoch"]),
                "parameters": {
                    name: np.asarray(model["parameters"][name], dtype=np.float32)
                    for name in PARAMETER_NAMES
                },
            }
        )

    return {
        "model_family": "numpy_two_hidden_layer_mlp_ensemble",
        "feature_columns": feature_columns,
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "config": {key: value for key, value in config.items()},
        "models": compact_models,
        "threshold": float(metrics["selected_threshold"]),
        "metrics": metrics,
        "feature_notes": {
            "TimeDays": "Original Time divided by 86400.",
            "TimeSinDay": "sin(2*pi*Time/86400) to capture daily periodicity.",
            "TimeCosDay": "cos(2*pi*Time/86400) to capture daily periodicity.",
            "LogAmount": "log1p(Amount) to reduce heavy-tail skew.",
            "AmountIsZero": "Indicator for zero-amount transactions.",
        },
    }


def predict_scores(values: pd.DataFrame, artifact: dict[str, object]) -> np.ndarray:
    feature_frame = build_feature_frame(values)
    feature_frame = feature_frame[artifact["feature_columns"]]

    mean = np.asarray(artifact["feature_mean"], dtype=np.float32)
    std = np.asarray(artifact["feature_std"], dtype=np.float32)
    X = transform_features(feature_frame, mean, std)

    negative_slope = float(artifact["config"]["negative_slope"])
    models = artifact["models"]
    score_stack = [
        predict_with_parameters(X, model["parameters"], negative_slope)
        for model in models
    ]
    return np.mean(np.vstack(score_stack), axis=0)


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)

    X_raw, y_series = load_dataset(data_path)
    feature_frame = build_feature_frame(X_raw)
    feature_columns = list(feature_frame.columns)
    y = y_series.to_numpy(dtype=np.int64)

    X_train_frame, X_val_frame, y_train, y_val = train_test_split(
        feature_frame,
        y,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=y,
    )

    train_mean, train_std = fit_standardization(X_train_frame)
    X_train = transform_features(X_train_frame[feature_columns], train_mean, train_std)
    X_val = transform_features(X_val_frame[feature_columns], train_mean, train_std)

    dummy_scores = np.full(y_val.shape[0], float(y_train.mean()), dtype=float)

    candidate_configs = build_candidate_configs(args.search_level)
    trial_summaries: list[dict[str, object]] = []

    print(f"Loaded {X_raw.shape[0]} transactions with {len(feature_columns)} engineered features.")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Validation rows: {X_val.shape[0]}")
    print(f"Fraud rate: {y.mean():.6f}")
    print(f"Trying {len(candidate_configs)} neural-network configurations.")
    print()

    for config_index, config in enumerate(candidate_configs, start=1):
        print(f"[Search {config_index}/{len(candidate_configs)}] {config['name']}")
        training_result = train_network(
            X_train,
            y_train,
            config=config,
            seed=args.seed,
            X_val=X_val,
            y_val=y_val,
            verbose=True,
        )
        trial_summary = summarize_trial(config, training_result, y_train, y_val)
        trial_summaries.append(trial_summary)
        val_metrics = trial_summary["validation_metrics_default_threshold"]
        print(
            "Validation ROC-AUC:",
            f"{float(val_metrics['roc_auc']):.6f}",
            "| Validation PR-AUC:",
            f"{float(val_metrics['pr_auc']):.6f}",
        )
        print()

    best_trial = choose_best_trial(trial_summaries)
    best_config = dict(best_trial["config"])

    print("Selected configuration:", best_config["name"])
    print(
        "Best validation ROC-AUC:",
        f"{float(best_trial['validation_metrics_default_threshold']['roc_auc']):.6f}",
    )
    print(
        "Best validation PR-AUC:",
        f"{float(best_trial['validation_metrics_default_threshold']['pr_auc']):.6f}",
    )
    print()

    ensemble_seeds = build_ensemble_seeds(args.seed, args.ensemble_size)
    holdout_models: list[dict[str, object]] = []

    for model_index, seed in enumerate(ensemble_seeds, start=1):
        print(f"[Ensemble {model_index}/{len(ensemble_seeds)}] seed={seed}")
        result = train_network(
            X_train,
            y_train,
            config=best_config,
            seed=seed,
            X_val=X_val,
            y_val=y_val,
            verbose=True,
        )
        holdout_models.append(
            {
                "seed": seed,
                "best_epoch": int(result["best_epoch"]),
                "parameters": result["parameters"],
            }
        )
        print()

    ensemble_validation_scores = average_model_scores(
        X_val,
        holdout_models,
        negative_slope=float(best_config["negative_slope"]),
    )
    ensemble_train_scores = average_model_scores(
        X_train,
        holdout_models,
        negative_slope=float(best_config["negative_slope"]),
    )
    selected_threshold, threshold_summary = find_best_threshold(y_val, ensemble_validation_scores)

    validation_metrics = {
        "dummy_validation": evaluate_scores(y_val, dummy_scores, threshold=0.5),
        "ensemble_train_default_threshold": evaluate_scores(y_train, ensemble_train_scores, threshold=0.5),
        "ensemble_validation_default_threshold": evaluate_scores(y_val, ensemble_validation_scores, threshold=0.5),
        "ensemble_validation_best_threshold": evaluate_scores(y_val, ensemble_validation_scores, threshold=selected_threshold),
        "threshold_search": threshold_summary,
    }

    final_mean = train_mean
    final_std = train_std
    final_models = holdout_models
    full_retrain_metrics: dict[str, object] | None = None

    if not args.skip_full_retrain:
        print("Retraining the selected ensemble on the full labeled dataset.")
        final_mean, final_std = fit_standardization(feature_frame[feature_columns])
        X_full = transform_features(feature_frame[feature_columns], final_mean, final_std)
        y_full = y.astype(np.int64)
        final_models = []

        for model_index, holdout_model in enumerate(holdout_models, start=1):
            print(
                f"[Full retrain {model_index}/{len(holdout_models)}] "
                f"seed={holdout_model['seed']} for {holdout_model['best_epoch']} epochs"
            )
            full_result = train_network(
                X_full,
                y_full,
                config=best_config,
                seed=int(holdout_model["seed"]),
                fixed_epochs=int(holdout_model["best_epoch"]),
                verbose=False,
            )
            final_models.append(
                {
                    "seed": int(holdout_model["seed"]),
                    "best_epoch": int(holdout_model["best_epoch"]),
                    "parameters": full_result["parameters"],
                }
            )

        final_training_scores = average_model_scores(
            X_full,
            final_models,
            negative_slope=float(best_config["negative_slope"]),
        )
        full_retrain_metrics = {
            "full_data_train_default_threshold": evaluate_scores(y_full, final_training_scores, threshold=0.5),
            "full_data_train_holdout_selected_threshold": evaluate_scores(
                y_full,
                final_training_scores,
                threshold=selected_threshold,
            ),
        }
        print()

    metrics = {
        "dataset": {
            "data_path": str(data_path),
            "rows": int(X_raw.shape[0]),
            "raw_features": int(X_raw.shape[1]),
            "engineered_features": len(feature_columns),
            "fraud_rate": float(y.mean()),
        },
        "split": {
            "random_state": int(args.seed),
            "validation_size": float(args.validation_size),
            "train_rows": int(X_train.shape[0]),
            "validation_rows": int(X_val.shape[0]),
            "train_fraud_rate": float(y_train.mean()),
            "validation_fraud_rate": float(y_val.mean()),
        },
        "search": {
            "search_level": args.search_level,
            "trials": trial_summaries,
            "selected_config_name": best_config["name"],
        },
        "validation": validation_metrics,
        "selected_threshold": float(selected_threshold),
        "full_retrain": full_retrain_metrics,
        "ensemble_size": int(args.ensemble_size),
    }

    artifact = build_prediction_artifact(
        feature_columns=feature_columns,
        mean=final_mean,
        std=final_std,
        models=final_models,
        config=best_config,
        metrics=metrics,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, args.model_out)
    args.metrics_out.write_text(json.dumps(to_serializable(metrics), indent=2), encoding="utf-8")

    print("Saved model artifact to:", args.model_out)
    print("Saved metrics JSON to:", args.metrics_out)
    print()
    print(
        "Ensemble validation ROC-AUC:",
        f"{float(validation_metrics['ensemble_validation_default_threshold']['roc_auc']):.6f}",
    )
    print(
        "Ensemble validation PR-AUC:",
        f"{float(validation_metrics['ensemble_validation_default_threshold']['pr_auc']):.6f}",
    )
    print("Selected validation threshold:", f"{selected_threshold:.6f}")


if __name__ == "__main__":
    main()
