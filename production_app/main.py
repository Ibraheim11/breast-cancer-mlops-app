from typing import Any

from production_app.config import settings
from production_app.data import load_dataset, split_data
from production_app.evaluation import evaluate_model
from production_app.logging_config import get_logger
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model

logger = get_logger(__name__)


def train_workflow() -> dict[str, Any]:
    """
    Run the complete training workflow.

    Workflow:
        Load data
        -> Split data
        -> Train pipeline
        -> Evaluate
        -> Save fitted pipeline

    Returns:
        Dictionary containing evaluation metrics.
    """
    logger.info("Starting training workflow.")

    X, y = load_dataset()

    logger.info(
        "Dataset loaded successfully: %d samples, %d features.",
        X.shape[0],
        X.shape[1],
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    logger.info(
        "Dataset split completed: %d training samples, %d testing samples.",
        len(X_train),
        len(X_test),
    )

    model = train_model(
        X_train,
        y_train,
    )

    logger.info("Model training completed successfully.")

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    logger.info(
        "Evaluation completed | accuracy=%.4f | precision=%.4f | recall=%.4f | f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )

    save_model(
        model,
        settings.model_path,
    )

    logger.info("Training workflow completed successfully.")

    return metrics


def predict_workflow(
    sample_index: int = 0,
) -> int:
    """
    Load the saved model and predict one dataset sample.

    No model retraining occurs in this workflow.

    Args:
        sample_index: Index of the sample to predict.

    Returns:
        Predicted class index.

    Raises:
        ValueError: If sample_index is outside the dataset range.
        FileNotFoundError: If the saved model does not exist.
    """
    logger.info(
        "Starting prediction workflow for sample index %d.",
        sample_index,
    )

    X, _ = load_dataset()

    if sample_index < 0 or sample_index >= len(X):
        logger.error(
            "Invalid sample index: %d.",
            sample_index,
        )

        raise ValueError(f"Sample index must be between 0 and {len(X) - 1}.")

    model = load_model(
        settings.model_path,
    )

    sample = X.iloc[[sample_index]]

    prediction = predict(
        model,
        sample,
    )

    predicted_class = int(prediction[0])

    logger.info(
        "Prediction workflow completed. Predicted class: %d.",
        predicted_class,
    )

    return predicted_class
