"""Module 1: Load and prepare train, validation, and test data."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file, target="Class", test_size=0.2, validation_size=0.2):
    data = pd.read_csv(file)
    X, y = data.drop(columns=target), data[target].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    X_subtrain, X_validation, y_subtrain, y_validation = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        stratify=y_train,
        random_state=42,
    )

    search_scaler = StandardScaler().fit(X_subtrain)
    X_subtrain = search_scaler.transform(X_subtrain).astype("float32")
    X_validation = search_scaler.transform(X_validation).astype("float32")

    final_scaler = StandardScaler().fit(X_train)
    X_train = final_scaler.transform(X_train).astype("float32")
    X_test = final_scaler.transform(X_test).astype("float32")

    return {
        "feature_columns": X.columns.tolist(),
        "X_subtrain": X_subtrain,
        "X_validation": X_validation,
        "y_subtrain": y_subtrain,
        "y_validation": y_validation,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": final_scaler,
    }
