from sklearn.preprocessing import StandardScaler


def create_preprocessor() -> StandardScaler:
    """
    Create the preprocessing transformer.

    Returns:
        StandardScaler instance.
    """
    return StandardScaler()
