"""
Centralised logging setup.
All modules get loggers via get_logger(__name__).
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Typically __name__ of the calling module.
    log_file : Path, optional
        If provided, also write to this file.
    """
    from src.config import CFG

    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times (Streamlit re-runs etc.)
    if logger.handlers:
        return logger

    logger.setLevel(CFG.logging.level)

    fmt = logging.Formatter(
        fmt=CFG.logging.format,
        datefmt=CFG.logging.datefmt,
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
