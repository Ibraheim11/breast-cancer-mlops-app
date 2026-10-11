from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from production_app.logging_config import get_logger


logger = get_logger(__name__)


def save_model(
    model: Pipeline,
    model_path: Path,
) -> None:
    """
    Save the fitted pipeline to disk.

    Args:
        model: Fitted machine learning pipeline.
        model_path: Destination path for the saved model.
    """
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    logger.info(
        "Model saved successfully to %s",
        model_path,
    )


def load_model(
    model_path: Path,
) -> Pipeline:
    """
    Load a previously saved pipeline from disk.

    Args:
        model_path: Path to the saved model.

    Returns:
        Loaded machine learning pipeline.

    Raises:
        FileNotFoundError: If the saved model does not exist.
    """
    if not model_path.exists():
        logger.error(
            "Saved model not found: %s",
            model_path,
        )

        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    model = joblib.load(model_path)

    logger.info(
        "Model loaded successfully from %s",
        model_path,
    )

    return model


def predict(
    model: Pipeline,
    input_data: pd.DataFrame,
):
    """
    Make predictions using a loaded pipeline.

    Args:
        model: Loaded fitted pipeline.
        input_data: Input features.

    Returns:
        Model predictions.

    Raises:
        ValueError: If prediction input is empty.
    """
    if input_data.empty:
        logger.error(
            "Prediction input is empty."
        )

        raise ValueError(
            "Prediction input cannot be empty."
        )

    predictions = model.predict(input_data)

    logger.info(
        "Prediction completed for %d sample(s).",
        len(input_data),
    )

    return predictions