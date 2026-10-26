"""
Assignment 03 - FastAPI Tests.

Verify:
1. Model/schema feature consistency.
2. Healthy model response.
3. Valid prediction.
4. Invalid input rejection.
5. Unavailable model handling.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from production_app import api
from production_app.data import load_dataset
from production_app.schemas import PredictionRequest


@pytest.fixture(autouse=True)
def clear_model_cache() -> Iterator[None]:
    """
    Clear the cached model before and after
    each test to keep tests independent.
    """

    api.get_prediction_model.cache_clear()

    yield

    api.get_prediction_model.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Create an in-process FastAPI test client."""

    with TestClient(api.app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload() -> dict[str, float]:
    """
    Build a valid request using real dataset values.

    The Pydantic schema determines which
    feature names are required.
    """

    X, _ = load_dataset()

    feature_names = [
        field.alias or name for name, field in PredictionRequest.model_fields.items()
    ]

    return {name: float(X.iloc[0][name]) for name in feature_names}


def test_schema_matches_saved_model() -> None:
    """
    Check that the API schema and final model
    use exactly the same 25 features.
    """

    model = api.get_prediction_model()

    model_features = list(model.feature_names_in_)

    schema_features = [
        field.alias or name for name, field in PredictionRequest.model_fields.items()
    ]

    assert len(schema_features) == 25

    assert schema_features == model_features


def test_health_success(client: TestClient) -> None:
    """GET /health must return HTTP 200."""

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
    }


def test_valid_prediction(
    client: TestClient,
    valid_payload: dict[str, float],
) -> None:
    """
    POST /predict with all 25 valid features
    must return a structured prediction.
    """

    response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert set(result) == {
        "predicted_class",
        "class_label",
    }

    assert result["predicted_class"] in (0, 1)

    expected_labels = {
        0: "malignant",
        1: "benign",
    }

    assert result["class_label"] == expected_labels[result["predicted_class"]]


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing",
        "extra",
        "malformed",
        "negative",
    ],
)
def test_invalid_prediction(
    client: TestClient,
    valid_payload: dict[str, float],
    invalid_case: str,
) -> None:
    """
    Invalid input must produce HTTP 422.

    Cases:
    - Missing feature
    - Extra feature
    - Malformed feature
    - Negative feature
    """

    payload = valid_payload.copy()

    first_feature = next(iter(payload))

    if invalid_case == "missing":
        del payload[first_feature]

    elif invalid_case == "extra":
        payload["unexpected feature"] = 10.0

    elif invalid_case == "malformed":
        payload[first_feature] = "invalid"

    elif invalid_case == "negative":
        payload[first_feature] = -1.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_model_unavailable(
    client: TestClient,
    valid_payload: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Simulate a missing model artifact.

    Verify:
    - Health returns HTTP 503.
    - Predict returns HTTP 503.
    - No internal traceback reaches clients.
    """

    def missing_model(_model_path: Path) -> None:
        raise FileNotFoundError("Simulated missing model for testing.")

    monkeypatch.setattr(
        api,
        "load_model",
        missing_model,
    )

    api.get_prediction_model.cache_clear()

    # Health should report unavailable.
    health_response = client.get("/health")

    assert health_response.status_code == 503

    assert health_response.json() == {
        "status": "unhealthy",
        "model_loaded": False,
    }

    # Prediction should also report unavailable.
    predict_response = client.post(
        "/predict",
        json=valid_payload,
    )

    assert predict_response.status_code == 503

    # Internal exception details must not
    # be exposed to the API consumer.
    assert "FileNotFoundError" not in predict_response.text

    assert "Traceback" not in predict_response.text
