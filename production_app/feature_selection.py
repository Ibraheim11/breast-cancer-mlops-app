"""
Feature selection utilities for Assignment 03.

Select features using training data only
to prevent data leakage.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif


def select_feature_names(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k: int = 10,
) -> list[str]:
    """
    Select the top K features using ANOVA F-test.

    Args:
        X_train: Training feature DataFrame.
        y_train: Training target labels.
        k: Number of features to select.

    Returns:
        List of selected feature names in their
        original dataset order.

    Raises:
        ValueError: If k is outside the valid range.
    """

    if not 1 <= k <= X_train.shape[1]:
        raise ValueError(f"k must be between 1 and {X_train.shape[1]}.")

    selector = SelectKBest(
        score_func=f_classif,
        k=k,
    )

    # IMPORTANT:
    # Fit the selector using training data only.
    selector.fit(X_train, y_train)

    selected_features = X_train.columns[selector.get_support()].tolist()

    return selected_features
