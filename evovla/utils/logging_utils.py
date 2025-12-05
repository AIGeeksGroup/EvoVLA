"""Logging utilities for EvoVLA."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Set up logger with console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class WandbLogger:
    """Wandb logging wrapper."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        project_name: str,
        run_name: Optional[str] = None,
        enabled: bool = True
    ):
        self.enabled = enabled
        if not enabled:
            return
        
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project_name,
                name=run_name,
                config=config,
                reinit=True
            )
        except ImportError:
            print("Warning: wandb not installed, logging disabled")
            self.enabled = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics."""
        if not self.enabled:
            return
        self.wandb.log(metrics, step=step)
    
    def finish(self):
        """Finish logging session."""
        if not self.enabled:
            return
        self.run.finish()

