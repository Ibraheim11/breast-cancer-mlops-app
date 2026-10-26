"""
Assignment 03 - Feature Selection Experiment.

Compare the original 30-feature baseline model
against a reduced-feature Logistic Regression model.

Feature selection is fitted on training data only.
The original saved model is never overwritten.
"""

import json
from typing import Any

from production_app.config import PROJECT_ROOT, settings
from production_app.data import load_dataset, split_data
from production_app.evaluation import evaluate_model
from production_app.feature_selection import select_feature_names
from production_app.logging_config import configure_logging, get_logger
from production_app.model_service import load_model, save_model
from production_app.training import train_model

logger = get_logger(__name__)

SELECTED_K = 10

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "breast_cancer_selected_pipeline.joblib"
)

SELECTED_FEATURES_PATH = PROJECT_ROOT / "models" / "selected_features.json"


def print_metrics(
    title: str,
    metrics: dict[str, Any],
) -> None:
    """Display model evaluation results."""

    print(f"\n{'=' * 50}")
    print(title)
    print("=" * 50)

    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1_score']:.4f}")

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


def main() -> None:
    """Compare baseline and selected-feature models."""

    configure_logging()

    logger.info("Starting feature selection experiment.")

    # -----------------------------------------
    # Step 1: Load dataset
    # -----------------------------------------

    X, y = load_dataset()

    logger.info(
        "Dataset loaded: %d samples, %d features.",
        X.shape[0],
        X.shape[1],
    )

    # -----------------------------------------
    # Step 2: Train/test split
    # -----------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    # -----------------------------------------
    # Step 3: Evaluate existing baseline
    # -----------------------------------------

    baseline_model = load_model(settings.model_path)

    if list(baseline_model.feature_names_in_) != list(X.columns):
        raise ValueError("Baseline model features do not match the original dataset.")

    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    print_metrics(
        "BASELINE MODEL - 30 FEATURES",
        baseline_metrics,
    )

    # -----------------------------------------
    # Step 4: Feature selection
    # -----------------------------------------

    selected_features = select_feature_names(
        X_train,
        y_train,
        k=SELECTED_K,
    )

    print("\nSelected Feature Names:")
    print("-" * 50)

    for index, feature in enumerate(selected_features, start=1):
        print(f"{index:02d}. {feature}")

    # -----------------------------------------
    # Step 5: Prepare selected features
    # -----------------------------------------

    X_train_selected = X_train.loc[
        :,
        selected_features,
    ]

    X_test_selected = X_test.loc[
        :,
        selected_features,
    ]

    logger.info(
        "Feature selection completed: %d -> %d features.",
        X.shape[1],
        len(selected_features),
    )

    # -----------------------------------------
    # Step 6: Train selected-feature model
    # -----------------------------------------

    selected_model = train_model(
        X_train_selected,
        y_train,
    )

    logger.info("Selected-feature model training completed.")

    # -----------------------------------------
    # Step 7: Evaluate selected model
    # -----------------------------------------

    selected_metrics = evaluate_model(
        selected_model,
        X_test_selected,
        y_test,
    )

    print_metrics(
        "SELECTED MODEL - 10 FEATURES",
        selected_metrics,
    )

    # -----------------------------------------
    # Step 8: Compare models
    # -----------------------------------------

    print("\n" + "=" * 50)
    print("BASELINE VS SELECTED MODEL")
    print("=" * 50)

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ]:
        baseline_value = baseline_metrics[metric]
        selected_value = selected_metrics[metric]

        difference = selected_value - baseline_value

        print(
            f"{metric:10s} | "
            f"Baseline: {baseline_value:.4f} | "
            f"Selected: {selected_value:.4f} | "
            f"Difference: {difference:+.4f}"
        )

    # -----------------------------------------
    # Step 9: Verify model feature order
    # -----------------------------------------

    model_features = list(selected_model.feature_names_in_)

    if model_features != selected_features:
        raise ValueError("Selected feature order does not match model input.")

    # -----------------------------------------
    # Step 10: Save candidate model
    # -----------------------------------------

    save_model(
        selected_model,
        CANDIDATE_MODEL_PATH,
    )

    # -----------------------------------------
    # Step 11: Save selected feature metadata
    # -----------------------------------------

    feature_metadata = {
        "method": "SelectKBest",
        "score_function": "f_classif",
        "k": len(selected_features),
        "feature_names": selected_features,
    }

    SELECTED_FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SELECTED_FEATURES_PATH.write_text(
        json.dumps(
            feature_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Candidate model and feature metadata saved.")

    print("\nFeature Selection Experiment Completed.")
    print(f"Candidate model: {CANDIDATE_MODEL_PATH}")
    print(f"Feature metadata: {SELECTED_FEATURES_PATH}")


if __name__ == "__main__":
    main()
