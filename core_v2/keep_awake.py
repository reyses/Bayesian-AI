"""Prevent Windows from sleeping during long-running processes."""

import ctypes
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

ES_CONTINUOUS       = 0x80000000
ES_SYSTEM_REQUIRED  = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


@contextmanager
def keep_awake(display: bool = False):
    """Context manager that prevents system sleep while active.

    Usage:
        with keep_awake():
            run_long_process()

    Args:
        display: Also prevent display from turning off (for live dashboard).
    """
    flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if display:
        flags |= ES_DISPLAY_REQUIRED
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
        logger.info("Sleep prevention: ENABLED")
    except Exception:
        logger.debug("Sleep prevention: not available (non-Windows)")
        yield
        return

    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        logger.info("Sleep prevention: DISABLED")
