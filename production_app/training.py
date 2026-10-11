import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from production_app.preprocessing import create_preprocessor


def create_pipeline() -> Pipeline:
    """
    Create the machine learning pipeline.

    Pipeline:
        StandardScaler -> LogisticRegression
    """
    return Pipeline(
        steps=[
            ("scaler", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                ),
            ),
        ]
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Train the machine learning pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.

    Returns:
        Fitted pipeline.
    """
    pipeline = create_pipeline()

    pipeline.fit(X_train, y_train)

    return pipeline
