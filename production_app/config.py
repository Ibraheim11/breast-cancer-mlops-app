import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """
    Application configuration settings.
    """

    app_env: str
    log_level: str
    model_path: Path
    test_size: float
    random_seed: int


def load_settings() -> Settings:
    """
    Load application settings from environment variables.

    Returns:
        Settings object containing runtime configuration.
    """
    model_path_value = os.getenv(
        "MODEL_PATH",
        "models/breast_cancer_pipeline.joblib",
    )

    model_path = Path(model_path_value)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    test_size = float(
        os.getenv(
            "TEST_SIZE",
            "0.20",
        )
    )

    if not 0 < test_size < 1:
        raise ValueError(
            "TEST_SIZE must be between 0 and 1."
        )

    return Settings(
        app_env=os.getenv(
            "APP_ENV",
            "development",
        ),
        log_level=os.getenv(
            "LOG_LEVEL",
            "INFO",
        ).upper(),
        model_path=model_path,
        test_size=test_size,
        random_seed=int(
            os.getenv(
                "RANDOM_SEED",
                "42",
            )
        ),
    )


settings = load_settings()