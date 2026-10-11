from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate the trained model using classification metrics.

    Args:
        model: Fitted machine learning pipeline.
        X_test: Testing features.
        y_test: Testing target.

    Returns:
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
        ),
        "f1_score": f1_score(
            y_test,
            y_pred,
            average="weighted",
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["malignant", "benign"],
        ),
    }

    return metrics
