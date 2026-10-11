import logging

from production_app.config import settings


LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def configure_logging() -> None:
    """
    Configure application-wide logging.
    """
    log_level = getattr(
        logging,
        settings.log_level,
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger.

    Args:
        name: Logger name.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)