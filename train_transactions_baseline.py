from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET_COLUMN = "Class"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models/transactions_logreg.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "artifacts/transactions_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a starter fraud-detection baseline for the transactions dataset."
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
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to store the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Where to store the training/validation metrics JSON.",
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


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    scaled_columns = [column for column in ["Time", "Amount"] if column in feature_columns]
    passthrough_columns = [column for column in feature_columns if column not in scaled_columns]

    transformers = []

    if scaled_columns:
        transformers.append(
            (
                "scaled_numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                scaled_columns,
            )
        )

    if passthrough_columns:
        transformers.append(
            (
                "passthrough_numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                passthrough_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_logistic_pipeline(feature_columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_columns)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def find_best_threshold(y_true: pd.Series, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)

    if thresholds.size == 0:
        return 0.5, {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_values))

    return float(thresholds[best_index]), {
        "f1": float(f1_values[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
    }


def evaluate_scores(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float | list[list[int]]]:
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


def as_float(value: float) -> float:
    return float(np.asarray(value).item())


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data_path)

    X, y = load_dataset(data_path)
    feature_columns = list(X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.validation_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    dummy_model = DummyClassifier(strategy="prior")
    dummy_model.fit(X_train, y_train)
    dummy_scores = dummy_model.predict_proba(X_val)[:, 1]

    logistic_model = build_logistic_pipeline(feature_columns)
    logistic_model.fit(X_train, y_train)

    train_scores = logistic_model.predict_proba(X_train)[:, 1]
    val_scores = logistic_model.predict_proba(X_val)[:, 1]
    best_threshold, threshold_summary = find_best_threshold(y_val, val_scores)

    metrics = {
        "dataset": {
            "data_path": str(data_path),
            "rows": int(X.shape[0]),
            "features": len(feature_columns),
            "fraud_rate": as_float(y.mean()),
        },
        "split": {
            "random_state": RANDOM_STATE,
            "validation_size": as_float(args.validation_size),
            "train_rows": int(X_train.shape[0]),
            "validation_rows": int(X_val.shape[0]),
            "train_fraud_rate": as_float(y_train.mean()),
            "validation_fraud_rate": as_float(y_val.mean()),
        },
        "dummy_validation": evaluate_scores(y_val, dummy_scores, threshold=0.5),
        "logistic_train_default_threshold": evaluate_scores(y_train, train_scores, threshold=0.5),
        "logistic_validation_default_threshold": evaluate_scores(y_val, val_scores, threshold=0.5),
        "logistic_validation_best_threshold": evaluate_scores(y_val, val_scores, threshold=best_threshold),
        "threshold_search": threshold_summary,
    }

    artifact = {
        "model": logistic_model,
        "feature_columns": feature_columns,
        "threshold": best_threshold,
        "metrics": metrics,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Saved model artifact to:", args.model_out)
    print("Saved metrics JSON to:", args.metrics_out)
    print()
    print("Validation ROC-AUC:", f"{metrics['logistic_validation_default_threshold']['roc_auc']:.6f}")
    print("Validation PR-AUC:", f"{metrics['logistic_validation_default_threshold']['pr_auc']:.6f}")
    print("Best validation threshold:", f"{best_threshold:.6f}")
    print("Best-threshold validation F1:", f"{metrics['logistic_validation_best_threshold']['f1']:.6f}")


if __name__ == "__main__":
    main()
