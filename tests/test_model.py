"""
Assignment 03 - Model Workflow Tests.

Verify:
1. Training and artifact generation.
2. Saved-model inference without retraining.
3. CLI prediction.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from production_app import main as workflow
from production_app.config import PROJECT_ROOT
from production_app.data import load_dataset
from production_app.model_service import load_model, predict


def test_training_workflow_saves_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Train the selected-feature model.

    Use temporary output paths so tests never
    overwrite the real final model.
    """

    temporary_model = tmp_path / "test_model.joblib"

    temporary_features = tmp_path / "test_features.json"

    monkeypatch.setattr(
        workflow,
        "FINAL_MODEL_PATH",
        temporary_model,
    )

    monkeypatch.setattr(
        workflow,
        "FINAL_FEATURES_PATH",
        temporary_features,
    )

    # Run the real training workflow.
    metrics = workflow.train_workflow()

    # Confirm that the files were created.
    assert temporary_model.exists()

    assert temporary_features.exists()

    # Confirm evaluation results exist.
    assert 0 <= metrics["accuracy"] <= 1

    assert 0 <= metrics["f1_score"] <= 1

    # Load feature metadata.
    metadata = json.loads(temporary_features.read_text(encoding="utf-8"))

    selected_features = metadata["feature_names"]

    # The final selected model must use 25 features.
    assert len(selected_features) == 25

    assert len(set(selected_features)) == 25

    # Load the newly trained artifact.
    model = load_model(temporary_model)

    # Model and metadata must have identical
    # feature names in identical order.
    assert list(model.feature_names_in_) == selected_features


def test_saved_model_predicts_without_retraining(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Load the existing final model and predict.

    Fail the test if Pipeline.fit() is called
    during inference.
    """

    model = load_model(workflow.FINAL_MODEL_PATH)

    X, _ = load_dataset()

    selected_features = list(model.feature_names_in_)

    assert len(selected_features) == 25

    sample = X.loc[[0], selected_features]

    def forbid_training(
        self: Pipeline,
        *args: object,
        **kwargs: object,
    ) -> None:
        raise AssertionError("Model fitting is not allowed during inference.")

    # Prevent accidental retraining.
    monkeypatch.setattr(
        Pipeline,
        "fit",
        forbid_training,
    )

    predictions = predict(model, sample)

    assert len(predictions) == 1

    assert int(predictions[0]) in (0, 1)


def test_cli_prediction() -> None:
    """
    Verify the original CLI prediction command
    still works with the final saved model.
    """

    result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "predict",
            "--sample-index",
            "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr

    assert "Prediction for sample 0:" in result.stdout
