import logging
import sys
import os
from pathlib import Path

def setup_logger(name, log_file, level=logging.DEBUG, console=False):
    """
    Sets up a logger with a file handler and optional console handler.
    Clears existing handlers to ensure clean logging in notebooks/interactive sessions.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    if log_file:
        try:
            # Ensure directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            fh = logging.FileHandler(log_file, mode='w') # Overwrite for new session
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Failed to setup file logging to {log_file}: {e}")

    # Console Handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Prevent propagation to root logger to avoid double logging if root is configured
    logger.propagate = False

    return logger
