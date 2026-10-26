"""
Assignment 03 - Cross-Validated Feature Selection.

Choose the number of features using training-only
cross-validation, then train and save a reduced-input
model without overwriting the original baseline.
"""

import json

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline

from production_app.config import PROJECT_ROOT, settings
from production_app.data import load_dataset, split_data
from production_app.evaluation import evaluate_model
from production_app.model_service import load_model, save_model
from production_app.training import train_model

CANDIDATE_K_VALUES = [8, 10, 12, 15, 20, 25]

MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_cv_selected_pipeline.joblib"

FEATURES_PATH = PROJECT_ROOT / "models" / "cv_selected_features.json"


def main() -> None:
    """Select features through training-only cross-validation."""

    print("=" * 60)
    print("ASSIGNMENT 03 - FEATURE SELECTION CROSS-VALIDATION")
    print("=" * 60)

    # 1. Load the original dataset.
    X, y = load_dataset()

    # 2. Use the original train/test split.
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    # 3. Load the original fitted baseline.
    baseline_model = load_model(settings.model_path)

    # Clone the model to preserve its configuration
    # without reusing fitted parameters during CV.
    baseline_template = clone(baseline_model)

    # 4. Define stratified cross-validation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=settings.random_seed,
    )

    # 5. Evaluate the baseline using training-only CV.
    baseline_cv_scores = cross_val_score(
        baseline_template,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )

    baseline_cv_mean = baseline_cv_scores.mean()

    print("\nBaseline Features:", X.shape[1])
    print(f"Baseline CV Weighted F1: {baseline_cv_mean:.4f}")

    # 6. Build a pipeline with the selector INSIDE CV.
    #
    # This is important because selection must be fitted
    # separately within each training fold.
    selection_pipeline = Pipeline(
        steps=[
            ("selector", SelectKBest(score_func=f_classif)),
            *[(name, clone(step)) for name, step in baseline_model.steps],
        ]
    )

    # 7. Compare different feature counts.
    search = GridSearchCV(
        estimator=selection_pipeline,
        param_grid={
            "selector__k": CANDIDATE_K_VALUES,
        },
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for params, mean_score, std_score in zip(
        search.cv_results_["params"],
        search.cv_results_["mean_test_score"],
        search.cv_results_["std_test_score"],
    ):
        k = params["selector__k"]

        print(
            f"K = {k:2d} | " f"Mean F1 = {mean_score:.4f} | " f"Std = {std_score:.4f}"
        )

    # 8. Retrieve the best feature count.
    best_k = search.best_params_["selector__k"]

    print("\n" + "=" * 60)
    print("BEST FEATURE SELECTION RESULT")
    print("=" * 60)

    print("Best K:", best_k)
    print(f"Best CV Weighted F1: {search.best_score_:.4f}")

    # 9. Extract selected feature names.
    #
    # GridSearchCV refits the selector using the full
    # training set after selecting the best K.
    best_selector = search.best_estimator_.named_steps["selector"]

    selected_features = X_train.columns[best_selector.get_support()].tolist()

    print("\nSelected Features:")

    for index, feature in enumerate(
        selected_features,
        start=1,
    ):
        print(f"{index:02d}. {feature}")

    # 10. Prepare reduced-input training/test data.
    X_train_selected = X_train.loc[:, selected_features]

    X_test_selected = X_test.loc[:, selected_features]

    # 11. Train a final candidate model.
    #
    # This model receives only the selected features.
    # The deployed API will NOT need the removed ones.
    selected_model = train_model(
        X_train_selected,
        y_train,
    )

    # 12. Verify exact feature order.
    if list(selected_model.feature_names_in_) != selected_features:
        raise ValueError("Model feature order does not match selected features.")

    # 13. Evaluate both models on the held-out test set.
    baseline_metrics = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    selected_metrics = evaluate_model(
        selected_model,
        X_test_selected,
        y_test,
    )

    print("\n" + "=" * 60)
    print("BASELINE VS CROSS-VALIDATED SELECTED MODEL")
    print("=" * 60)

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ):
        baseline_value = baseline_metrics[metric]
        selected_value = selected_metrics[metric]

        print(
            f"{metric:10s} | "
            f"Baseline: {baseline_value:.4f} | "
            f"Selected: {selected_value:.4f} | "
            f"Difference: {selected_value - baseline_value:+.4f}"
        )

    print("\nSelected Model Confusion Matrix:")
    print(selected_metrics["confusion_matrix"])

    # 14. Save a separate candidate artifact.
    #
    # The original 30-feature baseline is NOT overwritten.
    save_model(
        selected_model,
        MODEL_PATH,
    )

    # 15. Save feature metadata.
    metadata = {
        "method": "SelectKBest",
        "score_function": "f_classif",
        "selection_strategy": "training_only_5_fold_cv",
        "scoring": "f1_weighted",
        "original_feature_count": X.shape[1],
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "baseline_cv_f1": float(baseline_cv_mean),
        "selected_cv_f1": float(search.best_score_),
    }

    FEATURES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    FEATURES_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("Saved model:", MODEL_PATH)
    print("Saved feature metadata:", FEATURES_PATH)


if __name__ == "__main__":
    main()
