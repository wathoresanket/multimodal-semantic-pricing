"""
utils/logger.py  –  one-line setup, consistent format across all steps.

Usage:
    from utils.logger import get_logger
    log = get_logger("image_embeddings")
    log.info("Starting...")
"""
import logging
import sys
import os
from datetime import datetime


def get_logger(name: str, log_dir: str = None) -> logging.Logger:
    """
    Returns a logger that writes to BOTH:
      - stdout  (so `tail -f` on the terminal works)
      - logs/<name>_<timestamp>.log  (persistent file)
    """
    if log_dir is None:
        # resolve relative to project root regardless of cwd
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — always DEBUG level
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Stream handler — INFO+ to stdout (grep-friendly for nohup)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Logger initialised — writing to {log_file}")
    return logger
