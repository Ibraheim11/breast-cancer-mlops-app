from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    Validate the 30 input features required by
    the Breast Cancer classification pipeline.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    # Mean features
    mean_radius: NonNegativeFeature = Field(alias="mean radius")
    mean_texture: NonNegativeFeature = Field(alias="mean texture")
    mean_perimeter: NonNegativeFeature = Field(alias="mean perimeter")
    mean_area: NonNegativeFeature = Field(alias="mean area")
    mean_smoothness: NonNegativeFeature = Field(alias="mean smoothness")
    mean_compactness: NonNegativeFeature = Field(alias="mean compactness")
    mean_concavity: NonNegativeFeature = Field(alias="mean concavity")
    mean_concave_points: NonNegativeFeature = Field(alias="mean concave points")
    mean_symmetry: NonNegativeFeature = Field(alias="mean symmetry")
    mean_fractal_dimension: NonNegativeFeature = Field(alias="mean fractal dimension")

    # Error features
    radius_error: NonNegativeFeature = Field(alias="radius error")
    texture_error: NonNegativeFeature = Field(alias="texture error")
    perimeter_error: NonNegativeFeature = Field(alias="perimeter error")
    area_error: NonNegativeFeature = Field(alias="area error")
    smoothness_error: NonNegativeFeature = Field(alias="smoothness error")
    compactness_error: NonNegativeFeature = Field(alias="compactness error")
    concavity_error: NonNegativeFeature = Field(alias="concavity error")
    concave_points_error: NonNegativeFeature = Field(alias="concave points error")
    symmetry_error: NonNegativeFeature = Field(alias="symmetry error")
    fractal_dimension_error: NonNegativeFeature = Field(alias="fractal dimension error")

    # Worst features
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
