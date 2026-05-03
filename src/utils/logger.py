import logging
import sys
from pathlib import Path
import config


_LEVEL = logging.DEBUG if config.DEBUG else logging.INFO


_FORMATTER = logging.Formatter(
    fmt="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
    datefmt="%H:%M:%S"
)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with console output configured.

    Optionally writes to a log file if LOG_TO_FILE=true in .env.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)

        logger.info("Ingestion started")
        logger.debug("Chunk count: 47")
        logger.warning("Empty page skipped")
        logger.error("Failed to read PDF")

    Args:
        name: Logger name. Pass __name__ for module-level loggers.
              Shows exactly which file produced each log line.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # prevent duplicate handlers if get_logger called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(_LEVEL)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(_LEVEL)
    console_handler.setFormatter(_FORMATTER)
    logger.addHandler(console_handler)

    if config.LOG_TO_FILE: # type: ignore
        log_path = Path(config.LOG_FILE) # type: ignore
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(_LEVEL)
        file_handler.setFormatter(_FORMATTER)
        logger.addHandler(file_handler)

    # prevent log messages bubbling up to root logger
    logger.propagate = False

    return logger