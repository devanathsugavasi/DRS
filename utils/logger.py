"""
utils/logger.py — Structured logging for Cricket DRS
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from config.settings import LOGS_DIR


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a logger that writes structured records to both
    stdout and a rotating log file under data/logs/.
    """
    logger = logging.getLogger(name)
    if logger.handlers:            # avoid duplicate handlers on reimport
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Rotating file (10 MB × 5 backups)
    log_path = LOGS_DIR / f"{name}.log"
    fh = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger