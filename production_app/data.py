import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the Breast Cancer Wisconsin dataset.

    Returns:
        X: Feature DataFrame.
        y: Target Series.
    """
    dataset = load_breast_cancer(as_frame=True)

    X = dataset.data
    y = dataset.target

    return X, y

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.

    Stratification keeps the class distribution balanced
    between train and test sets.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )