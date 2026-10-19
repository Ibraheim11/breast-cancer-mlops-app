from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from sklearn.pipeline import Pipeline

from production_app.config import settings
from production_app.logging_config import configure_logging, get_logger
from production_app.model_service import load_model, predict
from production_app.schemas import PredictionRequest, PredictionResponse

# --------------------------------------------------
# Logging
# --------------------------------------------------

logger = get_logger(__name__)


# --------------------------------------------------
# Class Labels
# --------------------------------------------------

CLASS_LABELS = {
    0: "malignant",
    1: "benign",
}


# --------------------------------------------------
# Custom Exception
# --------------------------------------------------


class ModelUnavailableError(Exception):
    """Raised when the saved ML pipeline is unavailable."""


# --------------------------------------------------
# Model Loading
# --------------------------------------------------


@lru_cache(maxsize=1)
def get_prediction_model() -> Pipeline:
    """
    Load and cache the saved fitted ML pipeline.

    The model is loaded once per application process.
    No training or refitting occurs.
    """

    try:
        model = load_model(settings.model_path)

        # Validate feature names and their order.
        expected_features = [
            field.alias or name
            for name, field in PredictionRequest.model_fields.items()
        ]

        model_features = list(model.feature_names_in_)

        if expected_features != model_features:
            raise ValueError("API feature schema does not match the saved model.")

        logger.info("Saved ML pipeline is ready for inference.")

        return model

    except Exception as exc:
        logger.exception("Unable to initialize the saved ML pipeline.")

        raise ModelUnavailableError("Saved ML pipeline is unavailable.") from exc


# --------------------------------------------------
# Application Lifespan
# --------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Configure logging when the API starts.
    """

    configure_logging()

    logger.info("Starting Breast Cancer Prediction API.")

    yield

    logger.info("Stopping Breast Cancer Prediction API.")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="Breast Cancer Classification API",
    description=(
        "A validated Machine Learning prediction API "
        "built using FastAPI, Pydantic, and scikit-learn."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# GET /health
# --------------------------------------------------


@app.get(
    "/health",
    tags=["Health"],
)
def health_check() -> JSONResponse:
    """
    Check whether the API can load its saved ML pipeline.

    Returns:
        HTTP 200 when the model is available.
        HTTP 503 when the model is unavailable.
    """

    try:
        get_prediction_model()

    except ModelUnavailableError:
        logger.warning("Health check failed: model unavailable.")

        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
            },
        )

    logger.info("Health check completed successfully.")

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "model_loaded": True,
        },
    )


# --------------------------------------------------
# POST /predict
# --------------------------------------------------


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def make_prediction(
    request: PredictionRequest,
) -> PredictionResponse:
    """
    Validate one observation and make a prediction
    using the saved fitted ML pipeline.

    No model training occurs inside this endpoint.
    """

    logger.info("Received prediction request.")

    # Load the saved pipeline.
    try:
        model = get_prediction_model()

    except ModelUnavailableError as exc:
        logger.error("Prediction unavailable: saved model not ready.")

        raise HTTPException(
            status_code=503,
            detail="Prediction service is temporarily unavailable.",
        ) from exc

    # Convert validated Pydantic input into a DataFrame.
    input_data = request.model_dump(by_alias=True)

    model_features = list(model.feature_names_in_)

    input_df = pd.DataFrame(
        [input_data],
        columns=model_features,
    )

    # Run inference.
    try:
        predictions = predict(
            model,
            input_df,
        )

        predicted_class = int(predictions[0])

        class_label = CLASS_LABELS[predicted_class]

    except Exception as exc:
        logger.exception("Prediction execution failed.")

        raise HTTPException(
            status_code=500,
            detail="An internal prediction error occurred.",
        ) from exc

    logger.info(
        "Prediction completed successfully. Class: %d.",
        predicted_class,
    )

    return PredictionResponse(
        predicted_class=predicted_class,
        class_label=class_label,
    )
