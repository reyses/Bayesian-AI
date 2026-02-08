import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file, level=logging.DEBUG, console=False):
    """
    Sets up a logger with a file handler and optional console handler.
    Clears existing handlers to ensure clean logging in notebooks/interactive sessions.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to avoid duplicates and release file locks
    if logger.hasHandlers():
        for handler in list(logger.handlers):
            try:
                handler.close()
            except Exception:
                pass
            logger.removeHandler(handler)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler
    if log_file:
        try:
            # Ensure directory exists
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            # Remove existing log file to start fresh on new run
            if log_path.exists():
                try:
                    log_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete old log file {log_file}: {e}")

            fh = RotatingFileHandler(str(log_path), maxBytes=10*1024*1024, backupCount=1)
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
