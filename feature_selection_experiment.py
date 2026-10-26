"""Assignment 03: compare the 30-feature baseline with a 10-feature model.

SelectKBest is fitted on training data only. This experiment preserves the
original baseline and the final 25-feature model used by the API.
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

# Explicitly use the unchanged original 30-feature model as the baseline.
# settings.model_path now points to the final 25-feature API model, so it
# must not be used as the baseline for this experiment.
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_pipeline.joblib"

# Store experiment results separately. Never overwrite the baseline or the
# final selected-feature model used by FastAPI.
CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT / "models" / "breast_cancer_selected_pipeline.joblib"
)
SELECTED_FEATURES_PATH = PROJECT_ROOT / "models" / "selected_features.json"


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Print evaluation metrics and a confusion matrix."""
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
    """Train and evaluate a reduced-feature candidate against the baseline."""
    configure_logging()
    logger.info("Starting feature selection experiment.")

    # 1. Load data and split BEFORE fitting the feature selector.
    X, y = load_dataset()
    logger.info(
        "Dataset loaded: %d samples, %d features.",
        X.shape[0],
        X.shape[1],
    )
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    # 2. Load and verify the original 30-feature baseline artifact.
    baseline_model = load_model(BASELINE_MODEL_PATH)
    baseline_features = list(baseline_model.feature_names_in_)
    if baseline_features != list(X.columns):
        raise ValueError(
            "Baseline model features must match the original dataset "
            "names and order."
        )
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
    print_metrics("BASELINE MODEL - 30 FEATURES", baseline_metrics)

    # 3. Fit SelectKBest using ONLY the training split (no test leakage).
    selected_features = select_feature_names(
        X_train,
        y_train,
        k=SELECTED_K,
    )
    if len(selected_features) != SELECTED_K:
        raise ValueError("Selected feature count does not match k.")

    print("\nSelected Feature Names:")
    print("-" * 50)
    for index, feature in enumerate(selected_features, start=1):
        print(f"{index:02d}. {feature}")

    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]
    logger.info(
        "Feature selection completed: %d -> %d features.",
        X.shape[1],
        len(selected_features),
    )

    # 4. Train and evaluate the reduced-input candidate.
    selected_model = train_model(X_train_selected, y_train)
    logger.info("Selected-feature model training completed.")
    selected_metrics = evaluate_model(
        selected_model,
        X_test_selected,
        y_test,
    )
    print_metrics(
        f"SELECTED MODEL - {SELECTED_K} FEATURES",
        selected_metrics,
    )

    # 5. Compare on the SAME held-out test split.
    print("\n" + "=" * 50)
    print("BASELINE VS SELECTED MODEL")
    print("=" * 50)
    for metric in ("accuracy", "precision", "recall", "f1_score"):
        baseline_value = baseline_metrics[metric]
        selected_value = selected_metrics[metric]
        print(
            f"{metric:10s} | "
            f"Baseline: {baseline_value:.4f} | "
            f"Selected: {selected_value:.4f} | "
            f"Difference: {selected_value - baseline_value:+.4f}"
        )

    # 6. Preserve the exact feature names and their input order.
    if list(selected_model.feature_names_in_) != selected_features:
        raise ValueError("Selected feature order does not match the saved model input.")

    # 7. Save the experimental artifact and metadata ONLY.
    save_model(selected_model, CANDIDATE_MODEL_PATH)
    feature_metadata = {
        "method": "SelectKBest",
        "score_function": "f_classif",
        "k": len(selected_features),
        "feature_names": selected_features,
    }
    SELECTED_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTED_FEATURES_PATH.write_text(
        json.dumps(feature_metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Candidate model and feature metadata saved.")
    print("\nFeature Selection Experiment Completed.")
    print(f"Candidate model: {CANDIDATE_MODEL_PATH}")
    print(f"Feature metadata: {SELECTED_FEATURES_PATH}")


if __name__ == "__main__":
    main()
