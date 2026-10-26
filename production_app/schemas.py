"""
Pydantic schemas for Assignment 03.

Validate the exact 25 features required by the
final selected-feature machine learning model.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Every input feature must be a valid non-negative number.
NonNegativeFeature = Annotated[
    float,
    Field(
        strict=True,
        ge=0,
        allow_inf_nan=False,
    ),
]


class PredictionRequest(BaseModel):
    """
    Validate the 25 selected input features.

    Feature names and order must match the
    final saved machine learning pipeline.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # ---------------------------------
    # Mean Features (9)
    # ---------------------------------

    mean_radius: NonNegativeFeature = Field(alias="mean radius")

    mean_texture: NonNegativeFeature = Field(alias="mean texture")

    mean_perimeter: NonNegativeFeature = Field(alias="mean perimeter")

    mean_area: NonNegativeFeature = Field(alias="mean area")

    mean_smoothness: NonNegativeFeature = Field(alias="mean smoothness")

    mean_compactness: NonNegativeFeature = Field(alias="mean compactness")

    mean_concavity: NonNegativeFeature = Field(alias="mean concavity")

    mean_concave_points: NonNegativeFeature = Field(alias="mean concave points")

    mean_symmetry: NonNegativeFeature = Field(alias="mean symmetry")

    # ---------------------------------
    # Error Features (6)
    # ---------------------------------

    radius_error: NonNegativeFeature = Field(alias="radius error")

    perimeter_error: NonNegativeFeature = Field(alias="perimeter error")

    area_error: NonNegativeFeature = Field(alias="area error")

    compactness_error: NonNegativeFeature = Field(alias="compactness error")

    concavity_error: NonNegativeFeature = Field(alias="concavity error")

    concave_points_error: NonNegativeFeature = Field(alias="concave points error")

    # ---------------------------------
    # Worst Features (10)
    # ---------------------------------

    worst_radius: NonNegativeFeature = Field(alias="worst radius")

    worst_texture: NonNegativeFeature = Field(alias="worst texture")

    worst_perimeter: NonNegativeFeature = Field(alias="worst perimeter")

    worst_area: NonNegativeFeature = Field(alias="worst area")

    worst_smoothness: NonNegativeFeature = Field(alias="worst smoothness")

    worst_compactness: NonNegativeFeature = Field(alias="worst compactness")

    worst_concavity: NonNegativeFeature = Field(alias="worst concavity")

    worst_concave_points: NonNegativeFeature = Field(alias="worst concave points")

    worst_symmetry: NonNegativeFeature = Field(alias="worst symmetry")

    worst_fractal_dimension: NonNegativeFeature = Field(alias="worst fractal dimension")


class PredictionResponse(BaseModel):
    """
    Define the structured prediction response.
    """

    predicted_class: Literal[0, 1]

    class_label: Literal["malignant", "benign"]
