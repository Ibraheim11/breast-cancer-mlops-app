"""Assignment 03: cross-validated feature selection.

Choose the number of features using training-only cross-validation,
then save a reduced-input model without overwriting the 30-feature baseline.
"""

import json

from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from production_app.config import PROJECT_ROOT, settings
from production_app.data import load_dataset, split_data
from production_app.evaluation import evaluate_model
from production_app.model_service import load_model, save_model
from production_app.training import train_model

CANDIDATE_K_VALUES = [8, 10, 12, 15, 20, 25]

# Always load the original Assignment 01/02 baseline (30 features).
# settings.model_path now points to the Assignment 03 final model (25 features)
# and must NOT be used as the baseline in this comparison.
BASELINE_MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_pipeline.joblib"

# Save this experiment's results separately, without overwriting either
# the baseline artifact or the final API model.
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_cv_selected_pipeline.joblib"
FEATURES_PATH = PROJECT_ROOT / "models" / "cv_selected_features.json"


def main() -> None:
    """Run leakage-safe cross-validation, comparison, and artifact saving."""
    print("=" * 60)
    print("ASSIGNMENT 03 - FEATURE SELECTION CROSS-VALIDATION")
    print("=" * 60)

    # 1. Load and split before selecting features.
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=settings.test_size,
        random_state=settings.random_seed,
    )

    # 2. Load the unchanged 30-feature baseline explicitly.
    baseline_model = load_model(BASELINE_MODEL_PATH)
    baseline_features = list(baseline_model.feature_names_in_)
    if baseline_features != list(X.columns):
        raise ValueError(
            "Baseline model must use the original 30 dataset features in order."
        )

    # Clone keeps the baseline estimator settings but not its fitted state.
    baseline_template = clone(baseline_model)

    # 3. Use the same training-only CV strategy for baseline and candidates.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=settings.random_seed,
    )
    baseline_cv_scores = cross_val_score(
        baseline_template,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    baseline_cv_mean = float(baseline_cv_scores.mean())

    print("\nBaseline Features:", len(baseline_features))
    print(f"Baseline CV Weighted F1: {baseline_cv_mean:.4f}")

    # 4. Place feature selection INSIDE the CV pipeline to avoid leakage.
    selection_pipeline = Pipeline(
        steps=[
            ("selector", SelectKBest(score_func=f_classif)),
            *[(name, clone(step)) for name, step in baseline_model.steps],
        ]
    )
    search = GridSearchCV(
        estimator=selection_pipeline,
        param_grid={"selector__k": CANDIDATE_K_VALUES},
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

    best_k = int(search.best_params_["selector__k"])
    print("\n" + "=" * 60)
    print("BEST FEATURE SELECTION RESULT")
    print("=" * 60)
    print("Best K:", best_k)
    print(f"Best CV Weighted F1: {search.best_score_:.4f}")

    # GridSearchCV refitted the selector on the full training split.
    best_selector = search.best_estimator_.named_steps["selector"]
    selected_features = X_train.columns[best_selector.get_support()].tolist()
    if len(selected_features) != best_k:
        raise ValueError("Selected feature count does not match the chosen K.")

    print("\nSelected Features:")
    for index, feature in enumerate(selected_features, start=1):
        print(f"{index:02d}. {feature}")

    # 5. Final candidate training accepts ONLY the selected inputs.
    X_train_selected = X_train.loc[:, selected_features]
    X_test_selected = X_test.loc[:, selected_features]
    selected_model = train_model(X_train_selected, y_train)
    if list(selected_model.feature_names_in_) != selected_features:
        raise ValueError("Model feature order does not match selected features.")

    # 6. Use the held-out test set only for reporting, never choosing K.
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
    selected_metrics = evaluate_model(selected_model, X_test_selected, y_test)

    print("\n" + "=" * 60)
    print("BASELINE VS CROSS-VALIDATED SELECTED MODEL")
    print("=" * 60)
    for metric in ("accuracy", "precision", "recall", "f1_score"):
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

    # 7. Do not overwrite the baseline or the final serving artifact.
    save_model(selected_model, MODEL_PATH)
    metadata = {
        "method": "SelectKBest",
        "score_function": "f_classif",
        "selection_strategy": "training_only_5_fold_cv",
        "scoring": "f1_weighted",
        "original_feature_count": X.shape[1],
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "baseline_cv_f1": baseline_cv_mean,
        "selected_cv_f1": float(search.best_score_),
    }
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURES_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Saved model:", MODEL_PATH)
    print("Saved feature metadata:", FEATURES_PATH)


if __name__ == "__main__":
    main()
