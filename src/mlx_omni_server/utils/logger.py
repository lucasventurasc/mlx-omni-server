import logging
import os
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

# Create logs directory
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get project logger with optimized Rich configuration

    Args:
        name: Optional module name for the logger

    Returns:
        logging.Logger: Configured logger instance with Rich handler
    """
    # Create console with no file/line highlighting
    console = Console(highlight=False)

    # Configure Rich handler with custom settings
    rich_handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        enable_link_path=False,
        markup=True,
        rich_tracebacks=True,
        tracebacks_extra_lines=2,
        tracebacks_show_locals=True,
        log_time_format="%H:%M:%S.%f",  # Include microseconds for performance comparison
    )

    # Set log format to only include the message
    # Rich handler will add timestamps and log levels automatically
    FORMAT = "%(message)s"

    # Configure the root logger
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        handlers=[rich_handler],
    )

    # Get the named logger or use 'mlx_omni' as default
    logger_name = name if name else "mlx_omni"
    log = logging.getLogger(logger_name)

    return log


def set_logger_level(logger: logging.Logger, level: str):
    log_level = logging.getLevelNamesMapping().get(level.upper())
    logger.setLevel(log_level)


# Default logger
logger = get_logger()
