from rich.console import Console
from rich.logging import RichHandler
import logging


def setup_logging(logger_name: str) -> logging.Logger:
    """Configure and return a Rich logger"""
    console = Console()
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    return logging.getLogger(logger_name)
