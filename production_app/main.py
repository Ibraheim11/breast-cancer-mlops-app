"""
Training and prediction workflows.

Assignment 03:
Use the final 25 selected features while preserving
the original 30-feature baseline model.
"""

import json
from typing import Any

from production_app.config import PROJECT_ROOT, settings
from production_app.data import load_dataset, split_data
from production_app.evaluation import evaluate_model
from production_app.feature_selection import select_feature_names
from production_app.logging_config import get_logger
from production_app.model_service import load_model, predict, save_model
from production_app.training import train_model

logger = get_logger(__name__)

# Final feature count selected using training-only CV.
FINAL_FEATURE_COUNT = 25

# Keep the original baseline artifact unchanged.
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_final_pipeline.joblib"

FINAL_FEATURES_PATH = PROJECT_ROOT / "models" / "final_selected_features.json"


def train_workflow() -> dict[str, Any]:
    """
    Train and save the final selected-feature model.

    Steps:
        1. Load the dataset.
        2. Split into training and testing data.
        3. Select features using training data only.
        4. Train the fitted pipeline.
        5. Evaluate on the held-out test set.
        6. Save the model and selected feature names.

    The original baseline model is not overwritten.
    """

    logger.info("Starting Assignment 03 training workflow.")

    # Step 1: Load dataset.
    X, y = load_dataset()

    logger.info(
        "Dataset loaded: %d samples, %d features.",
        X.shape[0],
        X.shape[1],
    )

    # Step 2: Train/test split.
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    # Step 3: Select features using training data only.
    selected_features = select_feature_names(
        X_train,
        y_train,
        k=FINAL_FEATURE_COUNT,
    )

    logger.info(
        "Feature selection completed: %d -> %d features.",
        X.shape[1],
        len(selected_features),
    )

    # Step 4: Prepare reduced input datasets.
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]

    # Step 5: Train the final model.
    model = train_model(
        X_train_selected,
        y_train,
    )

    # Verify the exact feature names and order.
    if list(model.feature_names_in_) != selected_features:
        raise ValueError("Selected features do not match model input order.")

    logger.info("Final selected-feature model trained successfully.")

    # Step 6: Evaluate on held-out test data.
    metrics = evaluate_model(
        model,
        X_test_selected,
        y_test,
    )

    logger.info(
        "Evaluation | accuracy=%.4f | precision=%.4f " "| recall=%.4f | f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
    )

    # Step 7: Save the final fitted pipeline.
    save_model(
        model,
        FINAL_MODEL_PATH,
    )

    # Step 8: Save feature metadata.
    feature_metadata = {
        "method": "SelectKBest",
        "score_function": "f_classif",
        "selection_strategy": "training_only",
        "k": len(selected_features),
        "original_feature_count": X.shape[1],
        "selected_feature_count": len(selected_features),
        "feature_names": selected_features,
    }

    FINAL_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_FEATURES_PATH.write_text(
        json.dumps(feature_metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Final model and feature metadata saved.")
    logger.info("Training workflow completed successfully.")

    return metrics


def predict_workflow(
    sample_index: int = 0,
) -> int:
    """
    Load the saved final model and predict one sample.

    No retraining occurs during prediction.
    Only the final selected features are supplied.
    """

    logger.info(
        "Starting prediction for sample index %d.",
        sample_index,
    )

    # Step 1: Load original dataset.
    X, _ = load_dataset()

    if sample_index < 0 or sample_index >= len(X):
        raise ValueError(f"Sample index must be between 0 and {len(X) - 1}.")

    # Step 2: Load saved final model.
    model = load_model(FINAL_MODEL_PATH)

    # Step 3: Get required feature names in model order.
    model_features = list(model.feature_names_in_)

    if len(model_features) != FINAL_FEATURE_COUNT:
        raise ValueError("Saved model does not use the expected feature count.")

    missing_features = [
        feature for feature in model_features if feature not in X.columns
    ]

    if missing_features:
        raise ValueError("Saved model requires features missing from the dataset.")

    # Step 4: Select only required input features.
    sample = X.iloc[[sample_index]].loc[
        :,
        model_features,
    ]

    # Step 5: Predict without fitting the model.
    prediction = predict(
        model,
        sample,
    )

    predicted_class = int(prediction[0])

    logger.info(
        "Prediction completed. Class: %d.",
        predicted_class,
    )

    return predicted_class
